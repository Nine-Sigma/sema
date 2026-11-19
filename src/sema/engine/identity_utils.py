"""Identity predicates over the counts of `warehouse_lookup.RelationEvidence` (contract F4).

Each returns True when the predicate holds, False when it fails, and None when it cannot decide:
a count is missing because its statement timed out, or a key has a NULL part.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any


def one_smaller(columns: Sequence[str]) -> list[tuple[str, ...]]:
    """Each subset one column smaller; uniqueness is monotone, so these stand for every proper
    subset."""
    if len(columns) < 2:
        return []
    return list(combinations(columns, len(columns) - 1))


def key_holds(
    rows: int | None, distinct: int | None, null_rows: int | None,
    smaller_distincts: Sequence[int | None],
) -> bool | None:
    """Unique, minimal, and with no NULL part; `smaller_distincts` counts `one_smaller`'s sets."""
    if rows is None or distinct is None:
        return None
    if distinct != rows or rows in smaller_distincts:
        return False
    if null_rows is None or None in smaller_distincts:
        return None
    return True if null_rows == 0 else None


def entity_holds(
    rows: int | None, distinct: int | None, smaller_distincts: Sequence[int | None],
) -> bool | None:
    """Coarser than the row, and dropping any one key column lowers the distinct count."""
    if rows is None or distinct is None:
        return None
    if distinct >= rows or any(n is not None and n >= distinct for n in smaller_distincts):
        return False
    return None if None in smaller_distincts else True


def conflict_shares(groups: int, conflicting: Mapping[str, int]) -> dict[str, float]:
    """Each attribute's share of the key groups that hold more than one non-NULL value of it."""
    return {name: count / groups if groups else 0.0 for name, count in conflicting.items()}


DECLARED_KEY, SPEC_LIST, CONTAINED = "declared_key", "spec_list", "contained"
# F2's signal order: a lower rank outranks a higher one; no signal ranks last.
_SIGNAL_RANK = {DECLARED_KEY: 0, SPEC_LIST: 1, CONTAINED: 2}
PROVEN, SUPPORTED, OPEN, REJECTED = "proven", "supported", "open", "rejected"
ACCEPTED = (PROVEN, SUPPORTED)


@dataclass
class KeyFact:
    """One nominated column set on a relation: the key predicate's answer, its counts, and the
    signals that corroborate it."""

    columns: tuple[str, ...]
    holds: bool | None
    rows: int | None
    distinct: int | None
    null_rows: int | None
    signals: list[str] = field(default_factory=list)
    verification: str = OPEN


def signal_rank(signals: Sequence[str]) -> int:
    """A signal is its kind, optionally followed by ':' and what it names."""
    return min((_SIGNAL_RANK[s.split(":", 1)[0]] for s in signals), default=len(_SIGNAL_RANK))


def grade_keys(facts: Sequence[KeyFact]) -> KeyFact | None:
    """Grade every set; the key is the one passing set at the top signal rank, else None."""
    for fact in facts:
        if fact.holds is None:
            fact.verification = OPEN
        elif not fact.holds:
            fact.verification = REJECTED
        else:
            fact.verification = PROVEN if fact.signals else SUPPORTED
    passing = [f for f in facts if f.holds]
    if not passing:
        return None
    top = min(signal_rank(f.signals) for f in passing)
    tied = [f for f in passing if signal_rank(f.signals) == top]
    if len(tied) == 1:
        return tied[0]
    for fact in tied:
        fact.verification = OPEN
    return None


def key_payload(facts: Sequence[KeyFact], key: KeyFact | None) -> dict[str, Any]:
    """The relation's `has_key` payload: the key, else its first open set, else its first set;
    every other graded set is an alternative."""
    head = key or next((f for f in facts if f.verification == OPEN), facts[0])
    return {
        "columns": list(head.columns), "verification": head.verification,
        "evidence": {"rows": head.rows, "distinct": head.distinct,
                     "null_rows": head.null_rows, "signals": list(head.signals)},
        "alternatives": [{"columns": list(f.columns), "verification": f.verification}
                         for f in facts if f is not head],
    }
