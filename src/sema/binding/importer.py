"""Import a concept library into a target manifest plus an authored-evidence sidecar.

The manifest is the reviewable, versioned concept set; the sidecar holds the
library's existing bindings and value mappings as judge evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sema.binding.importer_utils import (
    Details,
    QueryFn,
    Record,
    RefKey,
    as_text,
    endpoint,
    endpoint_columns,
    expand_list,
    fetch,
    first_text,
    group_by_key,
    in_concept_scope,
    matched_details,
    normalize_bool,
    read_details,
    refs_payload,
    row_external_refs,
    select_sql,
    synonyms_of,
)
from sema.binding.models import (
    AuthoredColumnBinding,
    AuthoredEvidence,
    AuthoredValueMapping,
)
from sema.binding.spec import ConceptSourceSpec, ExternalRefRole
from sema.log import logger

MANIFEST_VERSION = 1
CONCEPT_SCHEME_KIND = "CONCEPT_SCHEME"


@dataclass(frozen=True)
class ConceptImport:
    manifest: dict[str, Any]
    authored: AuthoredEvidence


def import_concept_source(
    spec: ConceptSourceSpec, query: QueryFn, *, dataset_map: dict[str, str]
) -> ConceptImport:
    properties = _concept_properties(spec, query)
    vocabularies, terms = _value_sets(spec, query, properties)
    loaded = {(spec.identity_system, concept_id) for concept_id in properties}
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "descriptor": {
            "target_model_id": spec.id,
            "target_model_version": spec.release,
            "display_name": spec.id,
        },
        "vocabularies": vocabularies,
        "terms": terms,
        "entities": [_scheme_entity(spec, list(properties.values()))],
        "relationships": _relationships(spec, query, loaded),
    }
    authored = AuthoredEvidence(
        column_bindings=_column_bindings(spec, query, dataset_map),
        value_mappings=_value_mappings(spec, query),
    )
    logger.info(
        "Imported concept source {}: {} concepts, {} terms, {} relationships",
        spec.id,
        len(properties),
        len(terms),
        len(manifest["relationships"]),
    )
    return ConceptImport(manifest=manifest, authored=authored)


def write_import(
    result: ConceptImport, manifest_path: Path, evidence_path: Path
) -> None:
    manifest_path.write_text(yaml.safe_dump(result.manifest, sort_keys=False))
    evidence_path.write_text(result.authored.model_dump_json(indent=2))


def _scheme_entity(spec: ConceptSourceSpec, properties: list[Record]) -> Record:
    return {
        "qualified_name": spec.scheme.qualified_name,
        "kind": CONCEPT_SCHEME_KIND,
        "properties": properties,
        "context_card": {
            "card_version": spec.release,
            "description": spec.scheme.description,
        },
    }


def _concept_properties(spec: ConceptSourceSpec, query: QueryFn) -> dict[str, Record]:
    concepts, roles = spec.concepts, spec.concepts.roles
    join_column = spec.concept_details.on[0]
    rows = fetch(
        query,
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
        ],
        concepts.filter,
    )
    source_sql = select_sql(concepts.relation, [join_column], concepts.filter)
    details = read_details(query, spec.concept_details, source_sql)
    properties: dict[str, Record] = {}
    # One row per (concept, value): grouping stops one property per value.
    for group in group_by_key(rows, concepts.group_by).values():
        concept_id = first_text(group, roles.concept_id)
        if concept_id is not None:
            found = matched_details(group, join_column, details)
            properties[concept_id] = _concept_property(spec, concept_id, group, found)
    return dict(sorted(properties.items()))


def _concept_property(
    spec: ConceptSourceSpec, concept_id: str, group: list[Record], found: list[Details]
) -> Record:
    roles = spec.concepts.roles
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
    spec: ConceptSourceSpec, query: QueryFn, properties: dict[str, Record]
) -> tuple[list[Record], list[Record]]:
    roles, join_column = spec.value_sets.roles, spec.term_details.on[0]
    scope = in_concept_scope(roles.concept_id, spec)
    rows = fetch(
        query,
        spec.value_sets.relation,
        [
            roles.concept_id,
            roles.code,
            roles.display,
            roles.display_fallback,
            roles.vocabulary_id,
            roles.concept_code,
            roles.synonyms,
            join_column,
        ],
        scope,
    )
    source_sql = select_sql(spec.value_sets.relation, [join_column], scope)
    details = read_details(query, spec.term_details, source_sql)
    vocabularies: list[Record] = []
    terms: list[Record] = []
    for concept_id, group in sorted(group_by_key(rows, roles.concept_id).items()):
        if concept_id not in properties:
            continue
        vocabulary = f"VS_{concept_id}"
        properties[concept_id]["vocabulary_binding"] = {"vocabulary": vocabulary}
        vocabularies.append({"name": vocabulary, "source": "INLINE"})
        terms.extend(_terms(spec, vocabulary, group, details))
    return vocabularies, terms


def _terms(
    spec: ConceptSourceSpec,
    vocabulary: str,
    group: list[Record],
    details: dict[str, Details],
) -> list[Record]:
    roles = spec.value_sets.roles
    own_refs = _value_row_ref_roles(spec)
    terms: list[Record] = []
    for code, rows in sorted(group_by_key(group, roles.code).items()):
        found = matched_details(rows, spec.term_details.on[0], details)
        display = first_text(rows, roles.display) or first_text(
            rows, roles.display_fallback
        )
        display = display or code
        refs = {ref for d in found for ref in d.external_refs}
        refs |= {ref for row in rows for ref in row_external_refs(row, own_refs)}
        terms.append(
            {
                "vocabulary": vocabulary,
                "code": code,
                "display": display,
                "definition": next((d.definition for d in found if d.definition), None),
                "synonyms": sorted(synonyms_of(rows, roles.synonyms, []) - {display}),
                "external_refs": refs_payload(refs),
            }
        )
    return terms


def _value_row_ref_roles(spec: ConceptSourceSpec) -> list[ExternalRefRole]:
    roles = spec.value_sets.roles
    if roles.vocabulary_id is None or roles.concept_code is None:
        return []
    return [
        ExternalRefRole(system=f"{{{roles.vocabulary_id}}}", column=roles.concept_code)
    ]


def _relationships(
    spec: ConceptSourceSpec, query: QueryFn, loaded: set[RefKey]
) -> list[Record]:
    relation, roles = spec.relationships, spec.relationships.roles
    rows = fetch(
        query,
        relation.relation,
        [
            *endpoint_columns(roles.from_),
            *endpoint_columns(roles.to),
            roles.type,
            roles.hierarchical,
        ],
        _relationship_scope(spec),
    )
    kept: dict[tuple[RefKey, RefKey, str], Record] = {}
    for row in rows:
        start, end = endpoint(row, roles.from_), endpoint(row, roles.to)
        rel_type = as_text(row[roles.type])
        if start is None or end is None or rel_type is None:
            continue
        if start not in loaded and end not in loaded:
            continue
        kept[(start, end, rel_type)] = {
            "from": {"system": start[0], "id": start[1]},
            "to": {"system": end[0], "id": end[1]},
            "type": rel_type,
            "hierarchical": normalize_bool(row[roles.hierarchical])
            if roles.hierarchical
            else None,
            "far_end_loaded": start in loaded and end in loaded,
        }
    return [kept[key] for key in sorted(kept)]


def _relationship_scope(spec: ConceptSourceSpec) -> str:
    roles = spec.relationships.roles
    scope = (
        f"({in_concept_scope(roles.from_.id, spec)}"
        f" OR {in_concept_scope(roles.to.id, spec)})"
    )
    extra = spec.relationships.filter
    return f"({extra}) AND {scope}" if extra else scope


def _column_bindings(
    spec: ConceptSourceSpec, query: QueryFn, dataset_map: dict[str, str]
) -> list[AuthoredColumnBinding]:
    bindings_spec, roles = spec.authored_bindings, spec.authored_bindings.roles
    rows = fetch(
        query,
        bindings_spec.relation,
        [
            roles.concept_id,
            roles.dataset_label,
            roles.dataset_id,
            roles.columns,
            roles.project,
        ],
        in_concept_scope(roles.concept_id, spec),
    )
    bindings: list[AuthoredColumnBinding] = []
    for row in rows:
        concept_id = as_text(row[roles.concept_id])
        if concept_id is None:
            continue
        label, dataset_id = (
            as_text(row[roles.dataset_label]),
            as_text(row[roles.dataset_id]),
        )
        physical = _resolve_dataset(dataset_map, label, dataset_id)
        bindings.append(
            AuthoredColumnBinding(
                concept_ref=f"{spec.identity_system}:{concept_id}",
                dataset_label=label,
                dataset_id=dataset_id,
                physical=physical,
                columns=expand_list(row[roles.columns]),
                resolved=physical is not None,
                project=as_text(row[roles.project]) if roles.project else None,
            )
        )
    return bindings


def _resolve_dataset(
    dataset_map: dict[str, str], label: str | None, dataset_id: str | None
) -> str | None:
    for key in (label, dataset_id):
        if key is not None and key in dataset_map:
            return dataset_map[key]
    return None


def _value_mappings(
    spec: ConceptSourceSpec, query: QueryFn
) -> list[AuthoredValueMapping]:
    mappings_spec, roles = spec.value_mappings, spec.value_mappings.roles
    rows = fetch(
        query,
        mappings_spec.relation,
        [roles.concept_id, roles.source_value, roles.code, roles.status],
        in_concept_scope(roles.concept_id, spec),
    )
    approved_statuses = set(mappings_spec.approved_statuses)
    mappings: list[AuthoredValueMapping] = []
    for row in rows:
        concept_id = as_text(row[roles.concept_id])
        if concept_id is None:
            continue
        status = as_text(row[roles.status])
        mappings.append(
            AuthoredValueMapping(
                concept_ref=f"{spec.identity_system}:{concept_id}",
                source_value=as_text(row[roles.source_value]),
                codes=expand_list(row[roles.code]),
                status=status,
                approved=status in approved_statuses,
            )
        )
    return mappings
