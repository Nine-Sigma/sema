import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from showcase.ontology_query_benchmark import scorer

pytestmark = pytest.mark.unit

TOP_FIVE_GOLD = [
    {"value": "Oak, NOS", "count": "38956"},
    {"value": "Elm tree", "count": "32280"},
    {"value": "Upper branch, pine", "count": "30024"},
    {"value": "Birch bark", "count": "21922"},
    {"value": "Lower branch, pine", "count": "15521"},
]
TIERED_GOLD = [
    {"value": "Tier II", "count": "126746"},
    {"value": None, "count": "322871"},
    {"value": "Tier III", "count": "113305"},
]


def item(shape: str, gold: Any, qid: str = "QX", **extra: Any) -> dict[str, Any]:
    return {"id": qid, "result_shape": shape, "gold_answer": gold, "precision": 0,
            "tolerance": 0.0, "category_columns": [], **extra}  # fmt: skip


def strict(entry: dict[str, Any], rows: list[tuple[Any, ...]], complete: bool = True) -> bool:
    contract = scorer.contract_for(entry)
    return scorer.score(contract, rows, complete=complete).strict


def test_scalar_accepts_the_value_in_any_numeric_type() -> None:
    entry = item("scalar", ["42017"])
    assert strict(entry, [(42017,)])
    assert strict(entry, [("42017",)])
    assert strict(entry, [(Decimal("42017"),)])
    assert not strict(entry, [(939377,)])


def test_lucky_correct_count_in_an_extra_column_is_rejected() -> None:
    assert not strict(item("scalar", ["42017"]), [(42017, 12)])
    assert not strict(item("scalar", ["42017"]), [(42017,), (42017,)])


def test_multi_value_scalar_matches_within_tolerance_in_gold_numeric_order() -> None:
    entry = item("scalar", {"median_days": "41.0", "events": "153790"}, precision=None,
                 tolerance=1.0)  # fmt: skip
    assert strict(entry, [(41.8, 153790)])
    assert not strict(entry, [(42.5, 153790)])


def test_swapped_numeric_columns_get_no_strict_credit() -> None:
    entry = item("table_unordered", {"events": "341861", "people": "324530"}, precision=None)
    assert verdict(entry, [(341861, 324530)]) == (True, True)
    assert verdict(entry, [(324530, 341861)]) == (False, True)
    labelled = item("table_unordered", [{"stage": "Localized", "people": "32977",
                                         "deaths": "519"}])  # fmt: skip
    assert verdict(labelled, [(32977, 519, "Localized")]) == (True, True)
    assert verdict(labelled, [(519, 32977, "Localized")]) == (False, True)


def test_label_variants_fail_strict_in_both_directions() -> None:
    gold = item("table_unordered", [{"race": "Asian Indian", "n": "9"}], category_columns=["race"])
    assert verdict(gold, [("Asian", 9)])[0] is False
    white = item("table_unordered", [{"race": "White", "n": "9"}], category_columns=["race"])
    assert verdict(white, [("Non-Hispanic White", 9)])[0] is False


def test_tolerant_flag_accepts_within_one_percent_only() -> None:
    contract = scorer.contract_for(item("scalar", ["42017"]))
    near = scorer.score(contract, [(42000,)], complete=True)
    far = scorer.score(contract, [(40000,)], complete=True)
    assert (near.strict, near.tolerant) == (False, True)
    assert (far.strict, far.tolerant) == (False, False)


def verdict(entry: dict[str, Any], rows: list[tuple[Any, ...]]) -> tuple[bool, bool]:
    result = scorer.score(scorer.contract_for(entry), rows, complete=True)
    return result.strict, result.tolerant


def test_arbitrary_category_renaming_is_rejected() -> None:
    entry = item("table_unordered", TIERED_GOLD, category_columns=["value"])
    exact = [(" Tier II ", 126746), (None, 322871), ("Tier III", 113305)]
    assert verdict(entry, exact) == (True, True)
    recased = [("tier ii", 126746), (None, 322871), ("Tier III", 113305)]
    assert verdict(entry, recased) == (False, True)
    renamed = [("Category A", 126746), (None, 322871), ("Category B", 113305)]
    assert verdict(entry, renamed) == (False, False)
    swapped = [("Tier III", 126746), (None, 322871), ("Tier II", 113305)]
    assert verdict(entry, swapped) == (False, False)


def test_label_containment_counts_only_toward_the_tolerant_flag() -> None:
    gold = [{"age_group": "1: <50", "events": "22070"}, {"age_group": "2: 50-64", "events": "3"}]
    entry = item("table_unordered", gold, category_columns=["age_group"])
    assert verdict(entry, [("<50", 22070), ("50-64", 3)]) == (False, True)
    assert verdict(entry, [("under fifty", 22070), ("50-64", 3)]) == (False, False)
    race = item("table_unordered", [{"race": "Asian", "n": "72475"}], category_columns=["race"])
    assert verdict(race, [("Asian Indian", 72475)]) == (False, True)
    assert verdict(race, [("Asian", 72475)]) == (True, True)


def test_null_replaced_by_a_label_is_rejected() -> None:
    entry = item("table_unordered", TIERED_GOLD, category_columns=["value"])
    for label in ("No tier recorded", "", "NULL"):
        rows = [("Tier II", 126746), (label, 322871), ("Tier III", 113305)]
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


def test_a_ranked_top_five_keeps_gold_order() -> None:
    entry = item("ranked_list", TOP_FIVE_GOLD, qid="R08")
    rows = [(r["value"], int(r["count"])) for r in TOP_FIVE_GOLD]
    assert scorer.contract_for(entry).top_n == 5
    assert strict(entry, rows)
    assert strict(entry, [(n, v) for v, n in rows])
    assert strict(entry, [(i + 1, v, n) for i, (v, n) in enumerate(rows)])
    assert not strict(entry, [rows[1], rows[0], *rows[2:]])
    assert not strict(entry, [*rows, ("Oak", 15521)])
    assert not strict(entry, rows[:4])


def test_tied_rows_may_appear_in_either_order() -> None:
    gold = [{"v": "a", "n": "9"}, {"v": "b", "n": "5"}, {"v": "c", "n": "5"}]
    entry = item("ranked_list", gold)
    assert strict(entry, [("a", 9), ("c", 5), ("b", 5)])
    assert not strict(entry, [("b", 5), ("a", 9), ("c", 5)])


def test_declared_rank_column_defines_ties_and_is_optional_in_results() -> None:
    gold = [{"site": "Oak", "median": "407.0", "rnk": "1"},
            {"site": "Elm", "median": "403.0", "rnk": "1"}]  # fmt: skip
    entry = item("ranked_list", gold, precision=None)
    assert strict(entry, [("Elm", 403.0), ("Oak", 407.0)])
    assert strict(entry, [("Oak", 407.0, 1), ("Elm", 403.0, 1)])


def test_truncated_transport_gets_no_credit_even_with_the_right_value() -> None:
    contract = scorer.contract_for(item("scalar", ["42017"]))
    verdict = scorer.score(contract, [(42017,)], complete=False)
    assert (verdict.strict, verdict.tolerant, verdict.reason) == (False, False, "incomplete")


def test_retired_and_unknown_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="retired"):
        scorer.contract_for(item("scalar", ["1"], qid="R99"), frozenset({"R99"}))
    with pytest.raises(ValueError, match="result_shape"):
        scorer.contract_for(item("unsupported", None))


def test_load_contracts_reads_gold_and_refuses_a_retired_question(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"questions": [item("scalar", ["5"], qid="R01")]}))
    contracts = scorer.load_contracts([path], frozenset({"R99"}))
    assert scorer.score(contracts["R01"], [(5,)], complete=True).strict
    path.write_text(json.dumps({"questions": [item("scalar", ["5"], qid="R99")]}))
    with pytest.raises(ValueError, match="retired"):
        scorer.load_contracts([path], frozenset({"R99"}))


def strict_modes(entry: dict[str, Any], rows: list[tuple[Any, ...]]) -> tuple[bool, bool]:
    contract = scorer.contract_for(entry)
    ordered = scorer.score(contract, rows, complete=True)
    free = scorer.score(contract, rows, complete=True, order_free=True)
    return ordered.strict, free.strict


def test_order_free_mode_credits_swapped_measures_that_strict_rejects() -> None:
    entry = item("table_unordered", {"visits": "812", "members": "347"}, precision=None)
    assert strict_modes(entry, [(812, 347)]) == (True, True)
    assert strict_modes(entry, [(347, 812)]) == (False, True)


def test_a_wrong_value_fails_both_modes() -> None:
    entry = item("table_unordered", {"visits": "812", "members": "347"}, precision=None)
    assert strict_modes(entry, [(812, 348)]) == (False, False)
    assert strict_modes(entry, [(348, 812)]) == (False, False)


def test_order_free_mode_needs_one_column_mapping_for_every_row() -> None:
    gold = [{"branch": "North", "visits": "812", "members": "347"},
            {"branch": "South", "visits": "95", "members": "61"}]  # fmt: skip
    entry = item("table_unordered", gold, category_columns=["branch"])
    consistent = [(347, "North", 812), (61, "South", 95)]
    assert strict_modes(entry, consistent) == (False, True)
    per_row = [(347, "North", 812), (95, "South", 61)]
    assert strict_modes(entry, per_row) == (False, False)
