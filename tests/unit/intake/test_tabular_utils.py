"""Header detection: the first row that passes the three header tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _detect(rows: list[list[object]]) -> int | None:
    from sema.intake.loaders.tabular_utils import detect_header

    return detect_header(rows)


def test_header_on_row_3_after_a_title_block() -> None:
    rows: list[list[object]] = [
        ["Synthetic code library", None, None, None],
        [None, None, None, None],
        ["code", "name", "definition", "rank"],
        ["A", "Alpha", "First", 1],
        ["B", "Beta", "Second", 2],
    ]

    assert _detect(rows) == 3


def test_all_text_code_list_picks_row_1() -> None:
    rows: list[list[object]] = [["code", "label"]]
    rows += [[f"C{i % 3}", f"Label {i % 3}"] for i in range(10)]

    assert _detect(rows) == 1


def test_no_header_found_returns_none() -> None:
    rows: list[list[object]] = [[1, 2, 3], [4, 5, 6], [None, None, None]]

    assert _detect(rows) is None


def test_eighty_percent_non_empty_strings_is_the_boundary() -> None:
    four_of_five: list[list[object]] = [["a", "b", None, "d", "e"], [1, 2, None, 3, 4]]
    three_of_four: list[list[object]] = [["a", "b", None, "d"], [1, 2, None, 3]]

    assert _detect(four_of_five) == 1
    assert _detect(three_of_four) is None


def test_duplicate_strings_in_a_row_disqualify_it() -> None:
    rows: list[list[object]] = [
        ["name", "name", "kind"],
        ["code", "label", "kind_name"],
        ["A", "Alpha", "x"],
    ]

    assert _detect(rows) == 2


def test_value_recurring_in_its_column_disqualifies_the_row() -> None:
    rows: list[list[object]] = [
        ["A", "Alpha"],
        ["code", "label"],
        ["A", "Beta"],
    ]

    assert _detect(rows) == 2


def test_rows_past_the_scan_window_are_not_candidates() -> None:
    from sema.intake.loaders.tabular_utils import HEADER_SCAN_ROWS

    rows: list[list[object]] = [[i, i] for i in range(HEADER_SCAN_ROWS)]
    rows.append(["code", "label"])

    assert _detect(rows) is None


def test_numeric_text_is_not_a_header_string() -> None:
    rows: list[list[object]] = [["1", "2"], ["3", "4"]]

    assert _detect(rows) is None
