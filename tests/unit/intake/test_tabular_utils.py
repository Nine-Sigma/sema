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


def test_header_strings_must_be_unique_ignoring_case() -> None:
    rows: list[list[object]] = [["Code", "code", "label"], ["1", "2", "3"]]

    assert _detect(rows) is None


@pytest.mark.parametrize(
    "pair",
    [("a b", "a  b"), ("a b", "a\nb"), ("Line Break", "line \t break")],
)
def test_header_strings_must_be_unique_after_collapsing_whitespace(
    pair: tuple[str, str],
) -> None:
    rows: list[list[object]] = [[*pair, "label"], ["1", "2", "3"]]

    assert _detect(rows) is None


def test_header_strings_differing_in_more_than_whitespace_still_pass() -> None:
    rows: list[list[object]] = [["a b", "a  bc", "label"], ["1", "2", "3"]]

    assert _detect(rows) == 1


def test_trailing_empty_columns_do_not_count_toward_the_width() -> None:
    padding: list[object] = [None] * 14
    rows: list[list[object]] = [
        ["Revision history", None, None, None, *padding],
        ["Version", "Date", "Change", "Comments", *padding],
    ]
    rows += [[None] * 18 for _ in range(12)]

    assert _detect(rows) == 2


def test_a_column_filled_only_below_the_header_still_counts() -> None:
    rows: list[list[object]] = [
        ["code", "label", None, None, None],
        ["A", "Alpha", None, None, "x"],
    ]

    assert _detect(rows) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Additional Notes \n(where applicable)", "Additional Notes (where applicable)"),
        ("  code\t name  ", "code name"),
        ("include the  range", "include the range"),
        ("line\r\nbreak", "line break"),
        ("plain", "plain"),
    ],
)
def test_collapse_whitespace(text: str, expected: str) -> None:
    from sema.intake.loaders.tabular_utils import collapse_whitespace

    assert collapse_whitespace(text) == expected


def test_column_names_collapse_whitespace_in_header_text() -> None:
    from sema.intake.loaders.tabular_utils import column_names

    names = column_names(["Additional Notes \n(where applicable)", None, " a \t b "], 3)

    assert names == ["Additional Notes (where applicable)", "_empty_2", "a b"]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["1", "1.1", "1.2", "1.10"], "VARCHAR"),
        (["007", "008"], "VARCHAR"),
        (["+1", "2"], "VARCHAR"),
        (["1e5"], "VARCHAR"),
        (["12345678901234567890"], "VARCHAR"),
        (["0.5", "1.25", "2.0"], "DOUBLE"),
        (["1", "-2", " 3 "], "BIGINT"),
        ([1, 2.5], "DOUBLE"),
        ([2**63], "VARCHAR"),
    ],
)
def test_column_is_numeric_only_when_its_text_survives_the_round_trip(
    values: list[object], expected: str
) -> None:
    from sema.intake.loaders.tabular_utils import column_type

    assert column_type(values) == expected
