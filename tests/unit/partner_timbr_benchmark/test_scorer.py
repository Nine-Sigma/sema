import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from showcase.partner_timbr_benchmark import scorer

pytestmark = pytest.mark.unit

Q08_GOLD = [
    {"value": "Breast, NOS", "count": "38956"},
    {"value": "Prostate gland", "count": "32280"},
    {"value": "Upper lobe, lung", "count": "30024"},
    {"value": "Bone marrow", "count": "21922"},
    {"value": "Lower lobe, lung", "count": "15521"},
]
Q02_GOLD = [
    {"value": "Grade II", "count": "126746"},
    {"value": None, "count": "322871"},
    {"value": "Grade III", "count": "113305"},
]


def item(shape: str, gold: Any, qid: str = "QX", **extra: Any) -> dict[str, Any]:
    return {"id": qid, "result_shape": shape, "gold_answer": gold, "precision": 0,
            "tolerance": 0.0, "category_columns": [], **extra}  # fmt: skip


def strict(entry: dict[str, Any], rows: list[tuple[Any, ...]], complete: bool = True) -> bool:
    contract = scorer.contract_for(entry)
    return scorer.score(contract, rows, complete=complete).strict


def test_scalar_accepts_the_value_in_any_numeric_type() -> None:
    entry = item("scalar", ["939376"])
    assert strict(entry, [(939376,)])
    assert strict(entry, [("939376",)])
    assert strict(entry, [(Decimal("939376"),)])
    assert not strict(entry, [(939377,)])


def test_lucky_correct_count_in_an_extra_column_is_rejected() -> None:
    assert not strict(item("scalar", ["939376"]), [(939376, 12)])
    assert not strict(item("scalar", ["939376"]), [(939376,), (939376,)])


def test_multi_value_scalar_matches_in_any_column_order_within_tolerance() -> None:
    entry = item("scalar", {"median_days": "41.0", "tumors": "153790"}, precision=None,
                 tolerance=1.0)  # fmt: skip
    assert strict(entry, [(153790, 41.8)])
    assert not strict(entry, [(153790, 42.5)])


def test_tolerant_flag_accepts_within_one_percent_only() -> None:
    contract = scorer.contract_for(item("scalar", ["939376"]))
    near = scorer.score(contract, [(939000,)], complete=True)
    far = scorer.score(contract, [(900000,)], complete=True)
    assert (near.strict, near.tolerant) == (False, True)
    assert (far.strict, far.tolerant) == (False, False)


def verdict(entry: dict[str, Any], rows: list[tuple[Any, ...]]) -> tuple[bool, bool]:
    result = scorer.score(scorer.contract_for(entry), rows, complete=True)
    return result.strict, result.tolerant


def test_arbitrary_category_renaming_is_rejected() -> None:
    entry = item("table_unordered", Q02_GOLD, category_columns=["value"])
    exact = [(" Grade II ", 126746), (None, 322871), ("Grade III", 113305)]
    assert verdict(entry, exact) == (True, True)
    recased = [("grade ii", 126746), (None, 322871), ("Grade III", 113305)]
    assert verdict(entry, recased) == (False, True)
    renamed = [("Category A", 126746), (None, 322871), ("Category B", 113305)]
    assert verdict(entry, renamed) == (False, False)
    swapped = [("Grade III", 126746), (None, 322871), ("Grade II", 113305)]
    assert verdict(entry, swapped) == (False, False)


def test_label_containment_counts_only_toward_the_tolerant_flag() -> None:
    gold = [{"age_group": "1: <50", "tumors": "22070"}, {"age_group": "2: 50-64", "tumors": "3"}]
    entry = item("table_unordered", gold, category_columns=["age_group"])
    assert verdict(entry, [("<50", 22070), ("50-64", 3)]) == (False, True)
    assert verdict(entry, [("under fifty", 22070), ("50-64", 3)]) == (False, False)
    race = item("table_unordered", [{"race": "Asian", "n": "72475"}], category_columns=["race"])
    assert verdict(race, [("Asian Indian", 72475)]) == (False, True)
    assert verdict(race, [("Asian", 72475)]) == (True, True)


def test_null_replaced_by_a_label_is_rejected() -> None:
    entry = item("table_unordered", Q02_GOLD, category_columns=["value"])
    for label in ("No grade recorded", "", "NULL"):
        rows = [("Grade II", 126746), (label, 322871), ("Grade III", 113305)]
        assert not strict(entry, rows), label


def test_lexical_codes_are_not_collapsed_to_numbers() -> None:
    leading_zero = item("table_unordered", [{"code": "001", "n": "5"}])
    assert strict(leading_zero, [("001", 5)])
    assert not strict(leading_zero, [("1", 5)])
    assert not strict(leading_zero, [(1, 5)])
    plain = item("table_unordered", [{"code": "1", "n": "5"}])
    assert not strict(plain, [("001", 5)])
    assert strict(item("ranked_list", [{"g": "814", "n": "7"}]), [(814, 7)])


def test_bag_keeps_duplicates_and_set_ignores_them() -> None:
    bag = item("multiset", [["a"], ["a"], ["b"]], precision=None)
    assert strict(bag, [("a",), ("b",), ("a",)])
    assert not strict(bag, [("a",), ("b",)])
    table = item("table_unordered", [{"v": "x", "n": "1"}, {"v": "x", "n": "1"}])
    assert not strict(table, [("x", 1)])
    as_set = item("set", [["a"], ["b"]], precision=None)
    assert strict(as_set, [("a",), ("a",), ("b",)])
    assert not strict(as_set, [("a",)])


def test_q08_is_an_ordered_top_five() -> None:
    entry = item("ranked_list", Q08_GOLD, qid="Q08")
    rows = [(r["value"], int(r["count"])) for r in Q08_GOLD]
    assert scorer.contract_for(entry).top_n == 5
    assert strict(entry, rows)
    assert strict(entry, [(n, v) for v, n in rows])
    assert strict(entry, [(i + 1, v, n) for i, (v, n) in enumerate(rows)])
    assert not strict(entry, [rows[1], rows[0], *rows[2:]])
    assert not strict(entry, [*rows, ("Liver", 15521)])
    assert not strict(entry, rows[:4])


def test_tied_rows_may_appear_in_either_order() -> None:
    gold = [{"v": "a", "n": "9"}, {"v": "b", "n": "5"}, {"v": "c", "n": "5"}]
    entry = item("ranked_list", gold)
    assert strict(entry, [("a", 9), ("c", 5), ("b", 5)])
    assert not strict(entry, [("b", 5), ("a", 9), ("c", 5)])


def test_declared_rank_column_defines_ties_and_is_optional_in_results() -> None:
    gold = [{"site": "Liver", "median": "407.0", "rnk": "1"},
            {"site": "Lung", "median": "403.0", "rnk": "1"}]  # fmt: skip
    entry = item("ranked_list", gold, precision=None)
    assert strict(entry, [("Lung", 403.0), ("Liver", 407.0)])
    assert strict(entry, [("Liver", 407.0, 1), ("Lung", 403.0, 1)])


def test_truncated_transport_gets_no_credit_even_with_the_right_value() -> None:
    contract = scorer.contract_for(item("scalar", ["939376"]))
    verdict = scorer.score(contract, [(939376,)], complete=False)
    assert (verdict.strict, verdict.tolerant, verdict.reason) == (False, False, "incomplete")


def test_retired_and_unknown_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="retired"):
        scorer.contract_for(item("scalar", ["1"], qid="Q14"))
    with pytest.raises(ValueError, match="result_shape"):
        scorer.contract_for(item("unsupported", None))


def test_load_contracts_reads_gold_and_refuses_a_retired_question(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"questions": [item("scalar", ["5"], qid="Q01")]}))
    contracts = scorer.load_contracts([path])
    assert scorer.score(contracts["Q01"], [(5,)], complete=True).strict
    path.write_text(json.dumps({"questions": [item("scalar", ["5"], qid="Q14")]}))
    with pytest.raises(ValueError, match="retired"):
        scorer.load_contracts([path])
