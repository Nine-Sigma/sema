"""Relationship rows of the manifest: a relationships relation and hierarchy forms."""

from __future__ import annotations

from sema.binding.hierarchy_utils import (
    Pairs,
    parent_pairs_from_levels,
    parent_pairs_from_parent_column,
    parent_pairs_from_paths,
)
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
from sema.binding.spec import (
    ConceptSourceSpec,
    ConceptsSpec,
    HierarchySpec,
    LevelColumnsHierarchy,
    ParentColumnHierarchy,
    RelationshipsSpec,
)
from sema.intake.relations import RelationProvider
from sema.log import logger


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


def hierarchy_rows(
    spec: ConceptSourceSpec, provider: RelationProvider, loaded: set[RefKey]
) -> list[Record]:
    """Hierarchical relationship rows from a ``hierarchy`` declaration."""
    hierarchy, concepts = spec.hierarchy, spec.concepts
    if hierarchy is None or concepts is None:
        return []
    pairs, implicit = _hierarchy_pairs(hierarchy, concepts, provider)
    if implicit:
        logger.info(
            "Hierarchy of {}: {} implicit nodes get no edge: {}",
            spec.id,
            len(implicit),
            ", ".join(implicit),
        )
    system = spec.identity_system
    kept: dict[tuple[RefKey, RefKey], Record] = {}
    for child, parent in pairs:
        start, end = (system, child), (system, parent)
        if start in loaded or end in loaded:
            kept[(start, end)] = {
                "from": {"system": system, "id": child},
                "to": {"system": system, "id": parent},
                "type": hierarchy.relationship_type,
                "hierarchical": True,
                "far_end_loaded": start in loaded and end in loaded,
            }
    return [kept[key] for key in sorted(kept)]


def _hierarchy_pairs(
    hierarchy: HierarchySpec, concepts: ConceptsSpec, provider: RelationProvider
) -> tuple[Pairs, list[str]]:
    relation = hierarchy.relation or concepts.relation
    where = None if hierarchy.relation else concepts.filter
    if isinstance(hierarchy, ParentColumnHierarchy):
        child = hierarchy.child_column or concepts.roles.concept_id
        rows = fetch(provider, relation, [child, hierarchy.parent_column], where)
        return parent_pairs_from_parent_column(rows, child, hierarchy.parent_column)
    if isinstance(hierarchy, LevelColumnsHierarchy):
        rows = fetch(provider, relation, hierarchy.level_columns, where)
        return parent_pairs_from_levels(rows, hierarchy.level_columns)
    rows = fetch(provider, relation, [hierarchy.path_column], where)
    return parent_pairs_from_paths(rows, hierarchy.path_column, hierarchy.separator)
