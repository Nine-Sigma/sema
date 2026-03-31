"""Vocabulary edge materialization: CLASSIFIED_AS and IN_VOCABULARY.

Creates Vocabulary nodes and links them to Properties and Terms.
Extracted from materializer_utils.py to keep files under 400 lines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)
from sema.graph.loader_utils import (
    batch_create_classified_as,
    batch_create_in_vocabulary,
    batch_upsert_vocabularies,
)
from sema.models.physical_key import CanonicalRef

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader
    from sema.graph.materializer_utils import pick_winner as _pw


def _pick_winner_fn() -> Any:
    """Lazy import to avoid circular dependency."""
    from sema.graph.materializer_utils import pick_winner
    return pick_winner


def materialize_vocabulary_edges(
    loader: GraphLoader,
    groups: dict[tuple[str, str], list[Assertion]],
) -> None:
    """Materialize CLASSIFIED_AS and IN_VOCABULARY edges."""
    pick_winner = _pick_winner_fn()
    classified_edges: list[dict[str, Any]] = []
    vocab_names: set[str] = set()

    for (subj, pred), group in groups.items():
        if pred != AssertionPredicate.VOCABULARY_MATCH.value:
            continue
        winner = pick_winner(group)
        if not winner:
            continue
        vocab_name = winner.payload.get("value")
        if not vocab_name:
            continue
        vocab_names.add(vocab_name)

        try:
            pk = CanonicalRef.parse(subj)
            col_key = pk.column_key
            ds_id = pk.datasource_id
        except ValueError:
            continue
        if not col_key:
            continue

        classified_edges.append({
            "datasource_id": ds_id,
            "column_key": col_key,
            "vocabulary_name": vocab_name,
        })

    if vocab_names:
        batch_upsert_vocabularies(
            loader, [{"name": n} for n in vocab_names],
        )

    batch_create_classified_as(loader, classified_edges)

    in_vocab_edges = _collect_in_vocabulary_edges(groups)
    batch_create_in_vocabulary(loader, in_vocab_edges)


def _collect_in_vocabulary_edges(
    groups: dict[tuple[str, str], list[Assertion]],
) -> list[dict[str, Any]]:
    """Collect Term -> Vocabulary edges from decoded values."""
    from sema.graph.materializer_utils import pick_winner

    edges: list[dict[str, Any]] = []
    for (subj, pred), group in groups.items():
        if pred != AssertionPredicate.HAS_DECODED_VALUE.value:
            continue
        vocab_group = groups.get(
            (subj, AssertionPredicate.VOCABULARY_MATCH.value), [],
        )
        vocab_winner = pick_winner(vocab_group)
        if not vocab_winner:
            continue
        vocab_name = vocab_winner.payload.get("value")
        if not vocab_name:
            continue
        for a in group:
            if a.status in (
                AssertionStatus.REJECTED, AssertionStatus.SUPERSEDED,
            ):
                continue
            code = a.payload.get("raw", "")
            if code:
                edges.append({
                    "vocabulary_name": vocab_name, "code": code,
                })
    return edges
