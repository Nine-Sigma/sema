"""Vocabulary lifecycle: deprecate stale vocabularies after a build.

Lifecycle runs ONCE per build over the union of every table's active
vocabularies, not per table. Per-table deprecation lets a later table
deprecate vocabularies an earlier table introduced (finding D).

Deprecation is scoped to the schemas this build fully covered: a
vocabulary is deprecated only when every edge that references it lies
inside a covered schema. A sliced build, a failed table, or another
study's vocabularies are never deprecated by this run.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader

_INACTIVE = (AssertionStatus.REJECTED, AssertionStatus.SUPERSEDED)

_DEPRECATE_STALE_VOCABS = (
    "MATCH (v:Vocabulary) "
    "WHERE v.status = 'ACTIVE' "
    "AND NOT v.name IN $active_names "
    "AND EXISTS { "
    "MATCH (v)<-[r:IN_VOCABULARY|CLASSIFIED_AS]-() "
    "WHERE r.source_schema IN $schemas } "
    "AND NOT EXISTS { "
    "MATCH (v)<-[o:IN_VOCABULARY|CLASSIFIED_AS]-() "
    "WHERE o.source_schema IS NULL "
    "OR NOT o.source_schema IN $schemas } "
    "SET v.status = 'DEPRECATED'"
)


def active_vocab_names(assertions: list[Assertion]) -> set[str]:
    """Vocabulary names asserted ACTIVE by VOCABULARY_MATCH assertions."""
    names: set[str] = set()
    for a in assertions:
        if a.predicate != AssertionPredicate.VOCABULARY_MATCH:
            continue
        if a.status in _INACTIVE:
            continue
        name = a.payload.get("value")
        if name:
            names.add(name)
    return names


def covered_schemas(
    work_items: list[Any], results: list[Any],
) -> set[str]:
    """Schemas where every table in this build succeeded or was skipped.

    A failed table excludes its whole schema: its vocabularies never
    reached the active set, so absence proves nothing for that schema.
    """
    schema_by_ref = {wi.fqn: wi.schema for wi in work_items}
    failed = {
        schema_by_ref[r.table_ref]
        for r in results
        if r.status == "failed" and r.table_ref in schema_by_ref
    }
    return set(schema_by_ref.values()) - failed


def deprecate_stale_vocabularies(
    loader: GraphLoader, active_names: set[str], schemas: set[str],
) -> None:
    """Deprecate ACTIVE vocabularies absent from active_names.

    Only vocabularies whose every IN_VOCABULARY/CLASSIFIED_AS edge lies
    inside `schemas` qualify — a vocabulary another study still
    references keeps its status. No-op when active_names or schemas is
    empty: a build with zero vocabulary matches or zero fully-covered
    schemas must not deprecate anything.
    """
    if not active_names or not schemas:
        return
    loader._run(
        _DEPRECATE_STALE_VOCABS,
        active_names=sorted(active_names),
        schemas=sorted(schemas),
    )


def deprecate_stale_from_results(
    loader: GraphLoader,
    results: list[Any],
    work_items: list[Any],
    full_coverage: bool = True,
) -> None:
    """Deprecate stale vocabularies once over all tables' active set.

    Runs after every table finishes so the union of active vocabularies
    is known; a later table can never deprecate an earlier table's
    vocabulary (finding D). Each result carries `active_vocabularies`.

    Pass full_coverage=False for sliced or pattern-filtered builds:
    tables outside the slice share schemas with the slice, so absence
    from this run's active set proves nothing and nothing is deprecated.
    """
    if not full_coverage:
        return
    active: set[str] = set()
    for r in results:
        active.update(getattr(r, "active_vocabularies", None) or [])
    deprecate_stale_vocabularies(
        loader, active, covered_schemas(work_items, results),
    )
