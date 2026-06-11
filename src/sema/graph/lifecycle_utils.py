"""Vocabulary lifecycle: deprecate stale vocabularies after a build.

Lifecycle runs ONCE per build over the union of every table's active
vocabularies, not per table. Per-table deprecation lets a later table
deprecate vocabularies an earlier table introduced (finding D).
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


def deprecate_stale_vocabularies(
    loader: GraphLoader, active_names: set[str],
) -> None:
    """Deprecate ACTIVE vocabularies absent from active_names.

    No-op when active_names is empty: a build with zero vocabulary
    matches must not deprecate every existing vocabulary.
    """
    if not active_names:
        return
    loader._run(
        _DEPRECATE_STALE_VOCABS, active_names=sorted(active_names),
    )


def deprecate_stale_from_results(
    loader: GraphLoader, results: list[Any],
) -> None:
    """Deprecate stale vocabularies once over all tables' active set.

    Runs after every table finishes so the union of active vocabularies
    is known; a later table can never deprecate an earlier table's
    vocabulary (finding D). Each result carries `active_vocabularies`.
    """
    active: set[str] = set()
    for r in results:
        active.update(getattr(r, "active_vocabularies", None) or [])
    deprecate_stale_vocabularies(loader, active)
