"""Relationship rows of the manifest, from a relationships relation."""

from __future__ import annotations

from sema.binding.importer_utils import (
    Record,
    RefKey,
    as_text,
    endpoint,
    endpoint_columns,
    fetch,
    in_concept_scope,
    normalize_bool,
)
from sema.binding.spec import ConceptSourceSpec, RelationshipsSpec
from sema.intake.relations import RelationProvider


def relationship_rows(
    spec: ConceptSourceSpec,
    relation: RelationshipsSpec,
    provider: RelationProvider,
    loaded: set[RefKey],
) -> list[Record]:
    roles = relation.roles
    rows = fetch(
        provider,
        relation.relation,
        [
            *endpoint_columns(roles.from_),
            *endpoint_columns(roles.to),
            roles.type,
            roles.hierarchical,
        ],
        relationship_scope(spec, relation),
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


def relationship_scope(spec: ConceptSourceSpec, relation: RelationshipsSpec) -> str:
    roles = relation.roles
    scope = (
        f"({in_concept_scope(roles.from_.id, spec)}"
        f" OR {in_concept_scope(roles.to.id, spec)})"
    )
    extra = relation.filter
    return f"({extra}) AND {scope}" if extra else scope
