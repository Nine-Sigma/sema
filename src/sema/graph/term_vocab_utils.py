"""Resolve the vocabulary namespace for a column's :Term nodes.

Terms decoded from a column are namespaced by the column's matched
vocabulary when one exists, otherwise by the column's value-set name.
Both parent/child hierarchy edges and value-set membership for that
column must use the same namespace so they MATCH the same Term node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sema.graph.term_identity_utils import UNSCOPED_VOCAB
from sema.models.assertions import AssertionPredicate
from sema.models.physical_key import CanonicalRef

if TYPE_CHECKING:
    from sema.models.assertions import Assertion


def _pick_winner() -> Any:
    """Lazy import to avoid a materializer import cycle."""
    from sema.graph.materializer_utils import pick_winner

    return pick_winner


def resolve_term_vocab(
    col_ref: str,
    groups: dict[tuple[str, str], list[Assertion]],
    fallback: str,
) -> str:
    """Matched vocabulary for a column's terms, else ``fallback``."""
    pick_winner = _pick_winner()
    group = groups.get(
        (col_ref, AssertionPredicate.VOCABULARY_MATCH.value), [],
    )
    winner = pick_winner(group)
    if winner:
        return winner.payload.get("value") or fallback
    return fallback


def term_vocab_for_subject(
    subject_ref: str,
    groups: dict[tuple[str, str], list[Assertion]],
) -> str:
    """Term namespace for a column subject; sentinel if unparseable."""
    try:
        pk = CanonicalRef.parse(subject_ref)
    except ValueError:
        return UNSCOPED_VOCAB
    if not pk.column:
        return UNSCOPED_VOCAB
    return resolve_term_vocab(
        subject_ref, groups, f"{pk.table}_{pk.column}_values",
    )
