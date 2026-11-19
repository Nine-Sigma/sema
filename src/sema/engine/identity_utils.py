"""Identity predicates over the counts of `warehouse_lookup.RelationEvidence` (contract F4).

Each returns True when the predicate holds, False when it fails, and None when it cannot decide:
a count is missing because its statement timed out, or a key has a NULL part.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations


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
