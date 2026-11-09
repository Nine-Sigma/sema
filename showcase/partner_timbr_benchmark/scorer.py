"""Fresh typed scorer for the partner benchmark; the historical harness scorer stays frozen.

Each question's declared result_shape, category_columns, precision and tolerance become a
result contract. A result must have exactly the gold width (a ranked result may add one
rank column); columns may appear in any order. Gold cells that are canonical decimal text
compare as numbers, anything else (for example code '001') compares as exact text. NULL
equals only NULL, and '' is not NULL. Strict labels, category columns included, are equal
after trimming surrounding whitespace only. Ranked results keep gold order except among
tied rows and have exactly the gold top-N rows. Incomplete transport never scores.
Strict uses the declared tolerance. The secondary tolerant flag also accepts numbers
within 1% of gold and a category label that contains the gold wording or is contained in
it, never an arbitrary renaming.
"""

import itertools
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from showcase.partner_timbr_benchmark.scorer_utils import (
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
RETIRED = frozenset({"Q14"})
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

    @property
    def top_n(self) -> int | None:
        return len(self.gold) if self.kind == "ranked" else None


@dataclass(frozen=True)
class Verdict:
    strict: bool
    tolerant: bool
    reason: str


def contract_for(entry: Mapping[str, Any]) -> Contract:
    qid = str(entry["id"])
    if qid in RETIRED:
        raise ValueError(f"{qid} is retired")
    kind = SHAPE_KINDS.get(entry.get("result_shape", ""))
    if kind is None:
        raise ValueError(f"{qid} has no scorable result_shape {entry.get('result_shape')!r}")
    rows, names, ranks = gold_rows(entry["gold_answer"], entry["result_shape"], kind == "ranked")
    width = len(rows[0]) if rows else 0
    if not width or any(len(row) != width for row in rows):
        raise ValueError(f"{qid} gold rows are empty or ragged")
    columns = list(zip(*rows, strict=True))
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
                    Decimal(str(entry.get("tolerance") or 0)), ties)  # fmt: skip


def load_contracts(paths: Iterable[Path]) -> dict[str, Contract]:
    """Read question files (the only gold reader) into contracts keyed by question id."""
    contracts: dict[str, Contract] = {}
    for path in paths:
        raw = json.loads(Path(path).read_text())
        for entry in raw["questions"] if isinstance(raw, dict) else raw:
            contract = contract_for(entry)
            if contract.question_id in contracts:
                raise ValueError(f"duplicate question {contract.question_id}")
            contracts[contract.question_id] = contract
    return contracts


def score(contract: Contract, rows: Sequence[Sequence[Any]], *, complete: bool) -> Verdict:
    if not complete:
        return Verdict(False, False, "incomplete")
    if _matches(contract, rows, Decimal(0)):
        return Verdict(True, True, "match")
    tolerant = _matches(contract, rows, TOLERANT_SHARE)
    return Verdict(False, tolerant, "within_1pct" if tolerant else "mismatch")


def _precision(value: Any) -> int | None:
    if value is None or (isinstance(value, int | float) and float(value).is_integer()):
        return None if value is None else int(value)
    raise ValueError(f"precision {value!r} is not a number of decimal places")


def _matches(contract: Contract, rows: Sequence[Sequence[Any]], relative: Decimal) -> bool:
    width = len(contract.numeric)
    identity = [tuple(range(width))]
    orders: Iterable[tuple[int, ...]] = (
        itertools.permutations(range(width)) if width <= MAX_PERMUTED_WIDTH else identity
    )
    candidates = _width_candidates(contract, [tuple(r) for r in rows])
    return any(
        _compare(contract, [tuple(row[i] for i in order) for row in result], relative)
        for order in list(orders)
        for result in candidates
    )


def _width_candidates(contract: Contract, rows: list[Row]) -> list[list[Row]]:
    width = len(contract.numeric)
    widths = {len(row) for row in rows}
    if widths == {width}:
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
