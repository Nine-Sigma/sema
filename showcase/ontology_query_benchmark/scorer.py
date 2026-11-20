"""Fresh typed scorer for the benchmark; the historical harness scorer stays frozen.

Each question's declared result_shape, category_columns, precision and tolerance become a
result contract. A result must have exactly the gold width (a ranked result may add one rank
column). Columns may appear in any order, except that strict credit keeps the gold relative
order of numeric columns (swapped counts never pass strict); the order_free mode, a second
reported number, lifts only that numeric order rule. Gold cells that are canonical
decimal text compare as numbers, anything else (for example code '001') compares as exact
text. NULL equals only NULL, and '' is not NULL, unless the question's contract sets
null_equals_empty. An empty gold result scores when the entry declares its columns. Strict labels, category columns included,
are equal after trimming surrounding whitespace only. Ranked results keep gold order except
among tied rows and have exactly the gold top-N rows. Incomplete transport never scores.
Strict uses the declared tolerance. The secondary tolerant flag also accepts numbers within
1% of gold and a category label that contains the gold wording or is contained in it, never
an arbitrary renaming.
"""

import itertools
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from showcase.ontology_query_benchmark.scorer_utils import (
    Row,
    as_text,
    bijection,
    gold_rows,
    is_number_cell,
    labels_agree,
    numbers_agree,
    rank_like,
    tie_groups,
)

SHAPE_KINDS = {"scalar": "scalar", "set": "set", "multiset": "bag", "ranked_list": "ranked",
               "table_unordered": "table"}  # fmt: skip
TOLERANT_SHARE = Decimal("0.01")
MAX_PERMUTED_WIDTH = 6


@dataclass(frozen=True)
class Contract:
    question_id: str
    kind: str
    gold: tuple[Row, ...]
    numeric: tuple[bool, ...]
    category: tuple[bool, ...]
    precision: int | None
    tolerance: Decimal
    ties: tuple[int, ...] = ()
    null_equals_empty: bool = False

    @property
    def top_n(self) -> int | None:
        return len(self.gold) if self.kind == "ranked" else None


@dataclass(frozen=True)
class Verdict:
    strict: bool
    tolerant: bool
    reason: str


def contract_for(entry: Mapping[str, Any], retired: frozenset[str] = frozenset()) -> Contract:
    qid = str(entry["id"])
    if qid in retired:
        raise ValueError(f"{qid} is retired")
    kind = SHAPE_KINDS.get(entry.get("result_shape", ""))
    if kind is None:
        raise ValueError(f"{qid} has no scorable result_shape {entry.get('result_shape')!r}")
    rows, names, ranks = gold_rows(entry["gold_answer"], entry["result_shape"], kind == "ranked")
    # An empty expected result is scorable only when the entry declares its columns.
    width = len(rows[0]) if rows else len(entry.get("columns") or ())
    if not width or any(len(row) != width for row in rows):
        raise ValueError(f"{qid} gold rows are empty or ragged")
    columns = list(zip(*rows, strict=True)) or [()] * width
    numeric = tuple(
        any(c is not None for c in col) and all(c is None or is_number_cell(c) for c in col)
        for col in columns
    )
    declared = set(entry.get("category_columns") or ())
    category = tuple(
        bool(names) and names[i] in declared and not numeric[i] for i in range(width)
    )
    ties = tuple(tie_groups(rows, numeric, ranks)) if kind == "ranked" else ()
    return Contract(qid, kind, tuple(rows), numeric, category, _precision(entry.get("precision")),
                    Decimal(str(entry.get("tolerance") or 0)), ties,
                    bool(entry.get("null_equals_empty")))  # fmt: skip


def load_contracts(
    paths: Iterable[Path], retired: frozenset[str] = frozenset()
) -> dict[str, Contract]:
    """Read question files (the only gold reader) into contracts keyed by question id."""
    contracts: dict[str, Contract] = {}
    for path in paths:
        raw = json.loads(Path(path).read_text())
        for entry in raw["questions"] if isinstance(raw, dict) else raw:
            contract = contract_for(entry, retired)
            if contract.question_id in contracts:
                raise ValueError(f"duplicate question {contract.question_id}")
            contracts[contract.question_id] = contract
    return contracts


def score(
    contract: Contract, rows: Sequence[Sequence[Any]], *, complete: bool, order_free: bool = False
) -> Verdict:
    if not complete:
        return Verdict(False, False, "incomplete")
    if _matches(contract, rows, Decimal(0), keep_numeric_order=not order_free):
        return Verdict(True, True, "match")
    tolerant = _matches(contract, rows, TOLERANT_SHARE)
    return Verdict(False, tolerant, "within_1pct" if tolerant else "mismatch")


def _precision(value: Any) -> int | None:
    if value is None or (isinstance(value, int | float) and float(value).is_integer()):
        return None if value is None else int(value)
    raise ValueError(f"precision {value!r} is not a number of decimal places")


def _matches(
    contract: Contract,
    rows: Sequence[Sequence[Any]],
    relative: Decimal,
    *,
    keep_numeric_order: bool = True,
) -> bool:
    width = len(contract.numeric)
    identity = [tuple(range(width))]
    orders: Iterable[tuple[int, ...]] = (
        itertools.permutations(range(width)) if width <= MAX_PERMUTED_WIDTH else identity
    )
    if keep_numeric_order and not relative:
        orders = [o for o in orders if _numeric_order_kept(contract, o)]
    candidates = _width_candidates(contract, [tuple(r) for r in rows])
    return any(
        _compare(contract, [tuple(row[i] for i in order) for row in result], relative)
        for order in list(orders)
        for result in candidates
    )


def _numeric_order_kept(contract: Contract, order: tuple[int, ...]) -> bool:
    placed = [order[i] for i, numeric in enumerate(contract.numeric) if numeric]
    return placed == sorted(placed)


def _width_candidates(contract: Contract, rows: list[Row]) -> list[list[Row]]:
    width = len(contract.numeric)
    widths = {len(row) for row in rows}
    if not rows or widths == {width}:
        return [rows]
    if contract.kind != "ranked" or widths != {width + 1}:
        return []
    return [
        [row[:j] + row[j + 1 :] for row in rows]
        for j in range(width + 1)
        if rank_like([row[j] for row in rows])
    ]


def _compare(contract: Contract, result: list[Row], relative: Decimal) -> bool:
    def equal(gold: Row, found: Row) -> bool:
        return all(
            _cell_equal(contract, i, g, f, relative)
            for i, (g, f) in enumerate(zip(gold, found, strict=True))
        )

    gold = list(contract.gold)
    if contract.kind == "set":
        result = list(dict.fromkeys(result))
    if contract.kind != "ranked":
        return bijection(gold, result, equal)
    if len(result) != len(gold):
        return False
    groups = contract.ties
    spans = [[i for i, g in enumerate(groups) if g == group] for group in dict.fromkeys(groups)]
    return all(bijection([gold[i] for i in s], [result[i] for i in s], equal) for s in spans)


def _cell_equal(contract: Contract, column: int, gold: Any, found: Any, relative: Decimal) -> bool:
    if contract.null_equals_empty:
        gold, found = (None if cell == "" else cell for cell in (gold, found))
    if gold is None or found is None:
        return gold is None and found is None
    if contract.numeric[column]:
        return numbers_agree(gold, found, contract.precision, contract.tolerance, relative)
    text = as_text(found)
    if text is None:
        return False
    if contract.category[column] and relative:
        return labels_agree(str(gold), text)
    return text.strip() == str(gold).strip()
