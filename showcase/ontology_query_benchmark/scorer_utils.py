"""Cell typing, cell equality and row matching for the typed benchmark scorer."""

import datetime
import re
from collections.abc import Callable, Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

CANONICAL_NUMBER = re.compile(r"^-?(0|[1-9]\d*)(\.\d+)?$")
LABEL_TOKEN = re.compile(r"[0-9a-z]+")
RANK_KEYS = frozenset({"rank", "rnk"})

Row = tuple[Any, ...]
RowEqual = Callable[[Row, Row], bool]


def gold_rows(
    gold: Any, shape: str, drop_rank: bool
) -> tuple[list[Row], tuple[str, ...], list[Any] | None]:
    """Gold rows, the gold column names (empty when rows are lists) and declared ranks."""
    items = [gold] if shape == "scalar" or isinstance(gold, Mapping) else list(gold or ())
    rows: list[Row] = []
    names: tuple[str, ...] = ()
    ranks: list[Any] = []
    for entry in items:
        if isinstance(entry, Mapping):
            kept = {k: v for k, v in entry.items() if not (drop_rank and k.lower() in RANK_KEYS)}
            ranks += [v for k, v in entry.items() if drop_rank and k.lower() in RANK_KEYS]
            names = tuple(kept)
            rows.append(tuple(kept.values()))
        elif isinstance(entry, list | tuple):
            rows.append(tuple(entry))
        else:
            rows.append((entry,))
    return rows, names, (ranks if len(ranks) == len(rows) and ranks else None)


def is_number_cell(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float | Decimal):
        return True
    return isinstance(value, str) and bool(CANONICAL_NUMBER.match(value.strip()))


def as_number(value: Any) -> Decimal | None:
    if not is_number_cell(value):
        return None
    number = Decimal(repr(value)) if isinstance(value, float) else Decimal(str(value).strip())
    return number if number.is_finite() else None


def as_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, datetime.date):
        return value.isoformat()
    return None


def tie_groups(rows: list[Row], numeric: Sequence[bool], ranks: list[Any] | None) -> list[int]:
    """A group id per ranked gold row; consecutive rows with an equal rank key are tied."""
    keys = ranks if ranks is not None else [
        tuple(as_number(c) for c, n in zip(row, numeric, strict=True) if n) for row in rows
    ]  # fmt: skip
    groups: list[int] = []
    for i, key in enumerate(keys):
        groups.append(groups[-1] if i and key == keys[i - 1] else len(set(groups)))
    return groups


def labels_agree(gold: str, result: str) -> bool:
    """Author-worded category: one label's word tokens appear contiguously in the other."""
    g, r = LABEL_TOKEN.findall(gold.casefold()), LABEL_TOKEN.findall(result.casefold())
    return bool(g and r) and (_contains(g, r) or _contains(r, g))


def _contains(longer: list[str], shorter: list[str]) -> bool:
    n = len(shorter)
    return any(longer[i : i + n] == shorter for i in range(len(longer) - n + 1))


def numbers_agree(
    gold: Any, result: Any, precision: int | None, tolerance: Decimal, relative: Decimal
) -> bool:
    found = as_number(result)
    if found is None:
        return False
    expected = Decimal(str(gold).strip())
    if precision is not None:
        step = Decimal(1).scaleb(-precision)
        expected = expected.quantize(step, ROUND_HALF_UP)
        found = found.quantize(step, ROUND_HALF_UP)
    difference = abs(expected - found)
    return difference <= tolerance or difference <= relative * abs(expected)


def rank_like(values: Sequence[Any]) -> bool:
    numbers = [as_number(v) for v in values]
    if not numbers or any(n is None or n != n.to_integral_value() for n in numbers):
        return False
    ranks = [int(n) for n in numbers if n is not None]
    ordered = all(a <= b for a, b in zip(ranks, ranks[1:], strict=False))
    return ranks[0] == 1 and ordered and all(r <= i + 1 for i, r in enumerate(ranks))


def bijection(gold: Sequence[Row], result: Sequence[Row], equal: RowEqual) -> bool:
    """True when every gold row pairs with a distinct equal result row and none is left."""
    if len(gold) != len(result):
        return False
    edges = [[j for j, row in enumerate(result) if equal(g, row)] for g in gold]
    owner: dict[int, int] = {}

    def assign(i: int, seen: set[int]) -> bool:
        for j in edges[i]:
            if j not in seen:
                seen.add(j)
                if j not in owner or assign(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    return all(assign(i, set()) for i in range(len(gold)))
