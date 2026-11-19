"""The identity predicates over counts (contract F4): pure, so the build and the scorecard's
rechecks decide a fact the same way. None means a count is missing and the fact is not decided."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

pytestmark = pytest.mark.unit


def _utils() -> Any:
    return importlib.import_module("sema.engine.identity_utils")


def test_one_smaller_drops_each_column_once_and_keeps_the_order() -> None:
    assert _utils().one_smaller(["a", "b", "c"]) == [("a", "b"), ("a", "c"), ("b", "c")]
    assert _utils().one_smaller(["a"]) == []


@pytest.mark.parametrize(("rows", "distinct", "null_rows", "smaller", "holds"), [
    (6, 6, 0, [4, 3], True),
    (6, 5, 0, [4, 3], False),
    (6, 6, 0, [6, 3], False),
    (6, 6, 2, [4, 3], None),
    (6, None, 0, [4, 3], None),
    (6, 6, None, [4, 3], None),
    (6, 6, 0, [4, None], None),
    (6, 6, 0, [], True),
])
def test_a_key_is_unique_minimal_and_has_no_null_part(
    rows: int, distinct: int | None, null_rows: int | None, smaller: list[int | None],
    holds: bool | None,
) -> None:
    assert _utils().key_holds(rows, distinct, null_rows, smaller) is holds


@pytest.mark.parametrize(("rows", "distinct", "smaller", "holds"), [
    (6, 4, [2, 3], True),
    (6, 6, [2, 3], False),
    (6, 4, [4, 3], False),
    (6, 4, [], True),
    (None, 4, [2], None),
    (6, 4, [2, None], None),
])
def test_an_entity_is_coarser_than_the_row_and_needs_each_column(
    rows: int | None, distinct: int | None, smaller: list[int | None], holds: bool | None,
) -> None:
    assert _utils().entity_holds(rows, distinct, smaller) is holds


def test_conflict_share_is_each_attributes_share_of_the_groups() -> None:
    assert _utils().conflict_shares(4, {"x": 2, "y": 0}) == {"x": 0.5, "y": 0.0}
    assert _utils().conflict_shares(0, {"x": 0}) == {"x": 0.0}
