"""Import a concept library into a target manifest plus an authored-evidence sidecar.

The manifest is the reviewable, versioned concept set; the sidecar holds the
library's existing bindings and value mappings as judge evidence only, plus
SSSOM mappings and the retired concepts and terms the manifest omits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sema.binding.identifier_utils import card_version
from sema.binding.importer_entity_utils import (
    EntityTables,
    entity_records,
    entity_table_schemas,
)
from sema.binding.importer_evidence_utils import column_bindings, value_mappings
from sema.binding.importer_relationships_utils import hierarchy_rows, relationship_rows
from sema.binding.importer_terms_utils import (
    Collected,
    code_lists,
    collect_terms,
    store_terms,
    term_columns,
)
from sema.binding.importer_utils import (
    Details,
    Record,
    RefKey,
    fetch,
    first_text,
    group_by_key,
    in_concept_scope,
    matched_details,
    read_details,
    refs_payload,
    select_sql,
    synonyms_of,
)
from sema.binding.lifecycle_utils import lifecycle_columns, lifecycle_of, retirement_row
from sema.binding.mappings_utils import (
    SssomMapping,
    add_refs,
    exact_refs,
    read_mappings,
    subject_index,
)
from sema.binding.models import AuthoredEvidence
from sema.binding.spec import ConceptSourceSpec, ConceptsSpec, ValueSetsSpec
from sema.binding.term_staging import (
    clear_term_staging,
    staged_tables,
    write_term_staging,
)
from sema.ingest.duckdb_staging import Staging
from sema.intake.relations import RelationProvider
from sema.intake.table_schema import TableSchema
from sema.log import logger

MANIFEST_VERSION = 1
CONCEPT_SCHEME_KIND = "CONCEPT_SCHEME"


@dataclass(frozen=True)
class ConceptImport:
    """``table_schemas`` is keyed by the manifest entity qualified name."""

    manifest: dict[str, Any]
    authored: AuthoredEvidence
    table_schemas: dict[str, TableSchema] = field(default_factory=dict)


def import_concept_source(
    spec: ConceptSourceSpec,
    provider: RelationProvider,
    *,
    dataset_map: dict[str, str],
    staging: Staging | None = None,
) -> ConceptImport:
    provider.require(spec.relation_refs())
    _log_shape(spec)
    collected = Collected()
    if spec.concepts:
        _concept_properties(spec, spec.concepts, provider, collected)
    if spec.value_sets:
        _value_sets(spec, spec.value_sets, provider, collected, staging)
    code_lists(spec, provider, collected, staging)
    tables = (
        entity_table_schemas(spec, spec.entity_model, provider)
        if spec.entity_model
        else EntityTables()
    )
    loaded = _loaded_refs(spec, collected)
    mappings = _apply_mappings(spec, provider, collected, loaded)
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "descriptor": {
            "target_model_id": spec.id,
            "target_model_version": spec.release,
            "display_name": spec.id,
        },
        "vocabularies": collected.vocabularies,
        "terms": collected.terms,
        "entities": _scheme_entities(spec, list(collected.properties.values()))
        + entity_records(spec, tables),
        "relationships": _relationship_rows(spec, provider, loaded),
    }
    authored = _authored(spec, provider, dataset_map, mappings, collected)
    if staging is not None and collected.staged:
        write_term_staging(staging, spec.external_namespace, *staged_tables(collected.staged))
    elif staging is not None and spec.spec_version >= 2:
        clear_term_staging(staging, spec.external_namespace)
    _log_summary(spec, collected, manifest)
    return ConceptImport(
        manifest=manifest, authored=authored, table_schemas=tables.schemas
    )


def write_import(
    result: ConceptImport, manifest_path: Path, evidence_path: Path
) -> None:
    manifest_path.write_text(yaml.safe_dump(result.manifest, sort_keys=False))
    evidence_path.write_text(result.authored.model_dump_json(indent=2))


def _log_shape(spec: ConceptSourceSpec) -> None:
    for entry in spec.shape or []:
        logger.info(
            "Spec {} declares section {} as template {}",
            spec.id,
            entry.section,
            entry.template,
        )


def _log_summary(
    spec: ConceptSourceSpec, collected: Collected, manifest: dict[str, Any]
) -> None:
    logger.info(
        "Imported concept source {}: {} concepts, {} terms, {} relationships",
        spec.id,
        len(collected.properties),
        len(collected.terms),
        len(manifest["relationships"]),
    )
    kinds = [r.kind for r in collected.retirements]
    if kinds:
        logger.info(
            "Concept source {}: {} retired terms and {} retired concepts omitted "
            "from the manifest and kept in the sidecar retirements",
            spec.id,
            kinds.count("term"),
            kinds.count("concept"),
        )


def _authored(
    spec: ConceptSourceSpec,
    provider: RelationProvider,
    dataset_map: dict[str, str],
    mappings: list[SssomMapping],
    collected: Collected,
) -> AuthoredEvidence:
    bindings, value_maps = spec.authored_bindings, spec.value_mappings
    return AuthoredEvidence(
        column_bindings=column_bindings(spec, bindings, provider, dataset_map)
        if bindings
        else [],
        value_mappings=value_mappings(spec, value_maps, provider) if value_maps else [],
        mappings=mappings,
        retirements=collected.retirements,
    )


def _loaded_refs(spec: ConceptSourceSpec, collected: Collected) -> set[RefKey]:
    """Concept ids, and for version 2 also term codes, as relationship endpoints."""
    ids = set(collected.properties)
    if spec.spec_version >= 2:
        ids |= {term["code"] for term in collected.terms}
        ids |= {term.code for term in collected.staged}
    return {(spec.identity_system, loaded_id) for loaded_id in ids}


def _relationship_rows(
    spec: ConceptSourceSpec, provider: RelationProvider, loaded: set[RefKey]
) -> list[Record]:
    rows = (
        relationship_rows(spec, spec.relationships, provider, loaded)
        if spec.relationships
        else []
    )
    return rows + hierarchy_rows(spec, provider, loaded)


def _apply_mappings(
    spec: ConceptSourceSpec,
    provider: RelationProvider,
    collected: Collected,
    loaded: set[RefKey],
) -> list[SssomMapping]:
    if spec.mappings is None:
        return []
    loaded_ids = {loaded_id for _, loaded_id in loaded}
    loaded_ids |= {term["code"] for term in collected.terms}
    subjects = subject_index(spec.identity_system, loaded_ids)
    mappings = read_mappings(spec.mappings, provider, subjects)
    refs = exact_refs(mappings, spec.mappings.exact_predicates, subjects)
    for term in collected.terms:
        add_refs(term, refs.get(term["code"], set()))
    for concept_id, prop in collected.properties.items():
        add_refs(prop, refs.get(concept_id, set()))
    return mappings


def _scheme_entities(spec: ConceptSourceSpec, properties: list[Record]) -> list[Record]:
    if spec.scheme is None or (spec.concepts is None and not properties):
        return []
    return [
        {
            "qualified_name": spec.scheme.qualified_name,
            "kind": CONCEPT_SCHEME_KIND,
            "properties": properties,
            "context_card": {
                "card_version": card_version(spec.release),
                "description": spec.scheme.description,
            },
        }
    ]


def _concept_properties(
    spec: ConceptSourceSpec,
    concepts: ConceptsSpec,
    provider: RelationProvider,
    collected: Collected,
) -> None:
    roles = concepts.roles
    join_column = spec.concept_details.on[0] if spec.concept_details else None
    rows = fetch(
        provider,
        concepts.relation,
        [
            concepts.group_by,
            roles.concept_id,
            roles.name,
            roles.title,
            roles.definition,
            roles.data_type,
            roles.synonyms,
            join_column,
            *lifecycle_columns(roles),
        ],
        concepts.filter,
    )
    details: dict[str, Details] = {}
    if spec.concept_details and join_column:
        source_sql = select_sql(
            provider, concepts.relation, [join_column], concepts.filter
        )
        details = read_details(provider, spec.concept_details, source_sql)
    properties: dict[str, Record] = {}
    # One row per (concept, value): grouping stops one property per value.
    for group in group_by_key(rows, concepts.group_by).values():
        concept_id = first_text(group, roles.concept_id)
        if concept_id is None:
            continue
        lifecycle = lifecycle_of(group, roles, concepts.relation.name)
        if lifecycle.retired:
            collected.retirements.append(
                retirement_row("concept", None, concept_id, lifecycle)
            )
            continue
        found = matched_details(group, join_column, details)
        properties[concept_id] = _concept_property(spec, concepts, concept_id, group, found)
    collected.properties = dict(sorted(properties.items()))
    collected.retirements.sort(key=lambda r: r.code)


def _concept_property(
    spec: ConceptSourceSpec,
    concepts: ConceptsSpec,
    concept_id: str,
    group: list[Record],
    found: list[Details],
) -> Record:
    roles = concepts.roles
    name = first_text(group, roles.name) or concept_id
    definition = first_text(group, roles.definition) or next(
        (d.definition for d in found if d.definition), None
    )
    refs = {ref for d in found for ref in d.external_refs}
    synonyms = synonyms_of(group, roles.synonyms, [first_text(group, roles.title)]) - {
        name
    }
    return {
        "name": name,
        "type": first_text(group, roles.data_type) or "string",
        "synonyms": sorted(synonyms),
        "concept_ref": {
            "identity_system": spec.identity_system,
            "id": concept_id,
            "version": spec.release,
        },
        "definition": definition,
        "external_refs": refs_payload(refs),
    }


def _value_sets(
    spec: ConceptSourceSpec,
    value_sets: ValueSetsSpec,
    provider: RelationProvider,
    collected: Collected,
    staging: Staging | None,
) -> None:
    roles = value_sets.roles
    join_column = spec.term_details.on[0] if spec.term_details else None
    scope = in_concept_scope(provider, value_sets.relation, roles.concept_id, spec)
    rows = fetch(
        provider,
        value_sets.relation,
        [roles.concept_id, *term_columns(roles), join_column],
        scope,
    )
    details: dict[str, Details] = {}
    if spec.term_details and join_column:
        source_sql = select_sql(provider, value_sets.relation, [join_column], scope)
        details = read_details(provider, spec.term_details, source_sql)
    relation = value_sets.relation.name
    for concept_id, group in sorted(group_by_key(rows, roles.concept_id).items()):
        if concept_id not in collected.properties:
            continue
        vocabulary = f"VS_{concept_id}"
        terms = collect_terms(spec, roles, relation, vocabulary, group, details)
        collected.properties[concept_id]["vocabulary_binding"] = {"vocabulary": vocabulary}
        store_terms(spec, collected, vocabulary, terms, staging)

