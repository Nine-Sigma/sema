"""What the build keeps from intake's concept spec and its import: the stated domains, the code
lists and their roles, the reference columns, and each entity, declared join and inline list."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sema.binding.importer import import_concept_source
from sema.binding.spec import ConceptSourceSpec, load_concept_source_spec
from sema.engine.coded_column_utils import StatedDomain
from sema.engine.declared_join_utils import DeclaredJoin, DictionaryEntity, InlineList
from sema.ingest.duckdb_staging import Staging
from sema.intake.land.relations import RelationProvider


@dataclass(frozen=True)
class Dictionary:
    """What intake's concept spec states: its domains, each code list's codes and labels, its
    code and display role columns, each foreign key's reference columns, and the import's
    entities, declared joins and inline value lists."""

    spec_id: str
    domains: tuple[StatedDomain, ...]
    code_lists: dict[str, dict[str, str | None]]
    roles: dict[str, tuple[str, str | None]]
    reference_keys: list[tuple[str, ...]]
    entities: tuple[DictionaryEntity, ...] = ()
    declared_joins: tuple[DeclaredJoin, ...] = ()
    inline_lists: tuple[InlineList, ...] = ()


def read_dictionary(spec_path: str | None, staging_path: str | None) -> Dictionary | None:
    if spec_path is None:
        return None
    if staging_path is None:
        raise ValueError("concept_spec needs concept_staging, the file its relations live in")
    spec = _load_spec(Path(spec_path))
    with Staging(staging_path, schemas=()) as staging:
        provider = RelationProvider(warehouse=None, staging=staging)
        manifest = import_concept_source(spec, provider, dataset_map={}).manifest
    code_lists: dict[str, dict[str, str | None]] = {}
    for term in manifest["terms"]:
        code_lists.setdefault(term["vocabulary"], {})[str(term["code"])] = term.get("display")
    # shortcut: domains are read from per-file entries' `domains` maps only; read a role
    # column's domain too once a source states its domains that way.
    entries = (spec.entity_model.entities or []) if spec.entity_model else []
    domains = [
        StatedDomain(entry.entity, prop, vocabulary)
        for entry in entries for prop, vocabulary in entry.domains.items()
    ]
    roles: dict[str, tuple[str, str | None]] = {
        entry.vocabulary_id: (entry.roles.code, entry.roles.display)
        for entry in spec.code_lists or []
    }
    reference_keys = [
        tuple(key.reference_columns) for entry in entries for key in entry.foreign_keys
    ]
    entities, joins, inline_lists = _entity_record(manifest.get("entities") or [])
    for item in inline_lists:
        name = f"{item.entity}.{item.property}"
        code_lists[name] = dict(item.codes)
        domains.append(StatedDomain(item.entity, item.property, name))
    return Dictionary(spec.id, tuple(domains), code_lists, roles, reference_keys,
                      entities, joins, inline_lists)


def _load_spec(path: Path) -> ConceptSourceSpec:
    """The `concept_source` form, or the bare JSON form an intake run leaves."""
    text = path.read_text(encoding="utf-8")
    raw: Any = yaml.safe_load(text)
    if isinstance(raw, dict) and "concept_source" in raw:
        return load_concept_source_spec(path)
    return ConceptSourceSpec.model_validate_json(text)


def _entity_record(
    manifest_entities: list[dict[str, Any]],
) -> tuple[tuple[DictionaryEntity, ...], tuple[DeclaredJoin, ...], tuple[InlineList, ...]]:
    entities: list[DictionaryEntity] = []
    joins: list[DeclaredJoin] = []
    inline_lists: list[InlineList] = []
    for entity in manifest_entities:
        name = entity["qualified_name"]
        properties = entity.get("properties") or []
        entities.append(DictionaryEntity(name, tuple(p["name"] for p in properties)))
        joins += [
            DeclaredJoin(name, join["referenced_entity"],
                         tuple((local, remote) for local, remote in join["join_keys"]))
            for join in [*(entity.get("joins") or []),
                         *(entity.get("obligation") or {}).get("foreign_keys", [])]
        ]
        inline_lists += [
            InlineList(name, p["name"], dict(p["decoded_values"]))
            for p in properties if p.get("decoded_values")
        ]
    return tuple(entities), tuple(joins), tuple(inline_lists)
