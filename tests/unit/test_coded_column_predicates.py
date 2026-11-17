"""The containment predicate and the data-shape nominator for coded columns.

Both are properties of any source. `CONTAINMENT_FLOOR` is the least share of a column's
distinct non-null values, and of its non-null rows, that a value set must list before
other evidence can verify the binding: a code list may miss a legacy code, never most of
the column. `LOW_CARDINALITY_BOUND` is the most distinct values a column may hold and still
be nominated as coded by its data alone: past it the column is an identifier or a measure.
"""

from __future__ import annotations

import pytest

from sema.engine import join_detector_utils as detector
from sema.models.extraction import ExtractedColumn

pytestmark = pytest.mark.unit

SCHEMA = "s"
Profiles = dict[tuple[str, str, str], tuple[int, int]]


def test_the_floor_and_the_bound_are_fixed() -> None:
    assert detector.CONTAINMENT_FLOOR == 0.8
    assert detector.LOW_CARDINALITY_BOUND == 1000


def test_every_observed_value_listed() -> None:
    result = detector.containment({"A": 3, "B": 2, None: 4}, {"A", "B", "C"})
    assert result.observed_distinct == 2
    assert result.null_rows == 4
    assert result.listed_value_share == 1.0
    assert result.listed_row_share == 1.0
    assert result.holds


def test_an_unlisted_legacy_code_at_the_floor_still_holds() -> None:
    result = detector.containment({"A": 5, "B": 3, "C": 1, "D": 1, "X": 1}, {"A", "B", "C", "D"})
    assert result.listed_value_share == pytest.approx(4 / 5)
    assert result.listed_row_share == pytest.approx(10 / 11)
    assert result.holds


def test_unlisted_values_below_the_floor_do_not_hold() -> None:
    result = detector.containment({"A": 5, "B": 3, "C": 1, "X": 1, "Y": 1}, {"A", "B", "C"})
    assert result.listed_value_share == pytest.approx(3 / 5)
    assert not result.holds


def test_an_unlisted_value_holding_most_rows_does_not_hold() -> None:
    result = detector.containment({"A": 1, "B": 1, "C": 1, "D": 1, "X": 96}, {"A", "B", "C", "D"})
    assert result.listed_value_share == pytest.approx(4 / 5)
    assert result.listed_row_share == pytest.approx(4 / 100)
    assert not result.holds


def test_listed_codes_the_column_never_uses_leave_the_shares_alone() -> None:
    result = detector.containment({"A": 2}, {"A", "B", "C", "D", "E", "F"})
    assert result.listed_value_share == 1.0
    assert result.listed_row_share == 1.0
    assert result.holds


def test_a_column_of_nulls_has_no_shares_and_never_holds() -> None:
    result = detector.containment({None: 10}, {"A"})
    assert result.observed_distinct == 0
    assert result.null_rows == 10
    assert result.listed_value_share is None
    assert result.listed_row_share is None
    assert not result.holds


def _column(table: str, name: str, data_type: str = "VARCHAR") -> ExtractedColumn:
    return ExtractedColumn(
        name=name, table_name=table, catalog="c", schema=SCHEMA, data_type=data_type,
        ref=f"duckdb://c/{SCHEMA}/{table}/{name}",
    )


def _shape(
    *,
    coded_distinct: int = 3,
    key_rows: int = 4,
    key_distinct: int = 4,
    coded_type: str = "VARCHAR",
    label_column: bool = True,
) -> tuple[list[ExtractedColumn], Profiles]:
    """A fact relation with a coded column and a small code relation keyed by `kind_key`."""
    columns = [
        _column("fact", "fact_no", "INTEGER"),
        _column("fact", "kind", coded_type),
        _column("kinds", "kind_key"),
    ]
    profiles: Profiles = {
        (SCHEMA, "fact", "fact_no"): (5000, 5000),
        (SCHEMA, "fact", "kind"): (coded_distinct, 5000),
        (SCHEMA, "kinds", "kind_key"): (key_distinct, key_rows),
    }
    if label_column:
        columns.append(_column("kinds", "kind_title"))
        profiles[(SCHEMA, "kinds", "kind_title")] = (key_rows, key_rows)
    return columns, profiles


def _pairs(columns: list[ExtractedColumn], profiles: Profiles) -> set[tuple[str, ...]]:
    return {
        (c.fk_table, c.fk_column, c.pk_table, c.pk_column)
        for c in detector.nominate_by_data_shape(columns, profiles)
    }


NOMINATED = ("fact", "kind", "kinds", "kind_key")


def test_a_low_distinct_column_is_nominated_against_a_small_labelled_key_without_a_name() -> None:
    assert NOMINATED in _pairs(*_shape())


def test_the_bound_is_inclusive() -> None:
    assert NOMINATED in _pairs(*_shape(
        coded_distinct=detector.LOW_CARDINALITY_BOUND, key_rows=2000, key_distinct=2000,
    ))
    assert NOMINATED not in _pairs(*_shape(
        coded_distinct=detector.LOW_CARDINALITY_BOUND + 1, key_rows=2000, key_distinct=2000,
    ))


def test_a_key_with_duplicates_or_nulls_is_not_a_code_relation() -> None:
    assert NOMINATED not in _pairs(*_shape(key_rows=5, key_distinct=4))


def test_a_relation_with_no_text_column_beside_its_key_is_not_a_code_relation() -> None:
    assert NOMINATED not in _pairs(*_shape(label_column=False))


def test_incompatible_types_are_not_nominated() -> None:
    assert NOMINATED not in _pairs(*_shape(coded_type="INTEGER"))


def test_a_column_without_a_profile_is_not_nominated() -> None:
    columns, profiles = _shape()
    del profiles[(SCHEMA, "fact", "kind")]
    assert NOMINATED not in _pairs(columns, profiles)


def test_a_column_is_never_nominated_against_its_own_relation() -> None:
    columns, profiles = _shape()
    columns.append(_column("fact", "fact_title"))
    profiles[(SCHEMA, "fact", "fact_title")] = (5000, 5000)
    pairs = _pairs(columns, profiles)
    assert NOMINATED in pairs
    assert all(fk_table != pk_table for fk_table, _, pk_table, _ in pairs)
