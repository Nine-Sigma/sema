"""Entity identity compiles to one executable CTE (run in DuckDB, no warehouse)."""

from __future__ import annotations

from typing import Any

import duckdb
import pytest

from sema.compile.entity_identity import identity_cte_sql
from sema.models.target.identity import EntityIdentity

pytestmark = pytest.mark.unit

VISITS = "visit_id VARCHAR, site VARCHAR, seen_on INTEGER, ward VARCHAR, ward_kind VARCHAR"


def _resolve(identity: dict[str, Any], columns: str, rows: list[tuple[Any, ...]]) -> list[Any]:
    con = duckdb.connect()
    con.execute(f"CREATE TABLE src ({columns})")
    marks = ", ".join("?" for _ in columns.split(","))
    con.executemany(f"INSERT INTO src VALUES ({marks})", rows)
    cte = identity_cte_sql("entity", "src", EntityIdentity.model_validate(identity))
    return con.execute(f"{cte} SELECT * FROM entity ORDER BY ALL NULLS LAST").fetchall()


NULL_KEY_ROWS = [
    ("v1", "a", 1, "w1", "x"),
    ("v1", "a", 2, "w2", "x"),
    ("v1", None, 3, "w3", "x"),
    (None, "a", 4, "w4", "x"),
    (None, "a", 5, "w5", "x"),
]
FIRST_SEEN = [{"column": "seen_on", "rule": "min"}]


def test_null_keys_are_excluded() -> None:
    identity = {"key_columns": ["visit_id", "site"], "exclude_null_keys": True,
                "attributes": FIRST_SEEN}  # fmt: skip
    assert _resolve(identity, VISITS, NULL_KEY_ROWS) == [("v1", "a", 1)]


def test_null_keys_group_into_one_entity_per_key_value() -> None:
    identity = {"key_columns": ["visit_id", "site"], "attributes": FIRST_SEEN}
    assert _resolve(identity, VISITS, NULL_KEY_ROWS) == [
        ("v1", "a", 1), ("v1", None, 3), (None, "a", 4),
    ]


def test_each_row_with_a_null_key_is_its_own_entity() -> None:
    identity = {"key_columns": ["visit_id", "site"], "null_keys_group": False,
                "attributes": FIRST_SEEN}  # fmt: skip
    assert _resolve(identity, VISITS, NULL_KEY_ROWS) == [
        ("v1", "a", 1), ("v1", None, 3), (None, "a", 4), (None, "a", 5),
    ]


@pytest.mark.parametrize(("null_keys_group", "expected"), [
    (True, [("v1", "a", 1), ("v1", None, 3)]),
    (False, [("v1", "a", 1), ("v1", None, 3), ("v1", None, 6)]),
])  # fmt: skip
def test_only_the_named_key_columns_must_be_non_null(
    null_keys_group: bool, expected: list[tuple[Any, ...]]
) -> None:
    identity = {"key_columns": ["visit_id", "site"], "exclude_null_keys": ["visit_id"],
                "null_keys_group": null_keys_group, "attributes": FIRST_SEEN}  # fmt: skip
    rows = [*NULL_KEY_ROWS, ("v1", None, 6, "w6", "x")]
    assert _resolve(identity, VISITS, rows) == expected


def test_ranked_preferred_values_win_in_order_before_the_rule() -> None:
    identity = {"key_columns": ["visit_id"], "attributes": [
        {"column": "ward", "rule": "min_by", "by": "seen_on",
         "prefer": {"column": "ward_kind", "value": ["primary", "secondary"]}},
    ]}  # fmt: skip
    rows = [
        ("v1", "a", 1, "w1", "other"), ("v1", "a", 4, "w4", "secondary"),
        ("v1", "a", 5, "w5", "primary"),
        ("v2", "a", 1, "w1", "other"), ("v2", "a", 3, "w3", "secondary"),
        ("v2", "a", 2, "w2", "secondary"),
        ("v3", "a", 2, "w2", None), ("v3", "a", 1, "w1", "other"),
    ]
    assert _resolve(identity, VISITS, rows) == [("v1", "w5"), ("v2", "w2"), ("v3", "w1")]


def test_min_and_max_ignore_null_values() -> None:
    identity = {"key_columns": ["visit_id"], "attributes": [
        {"column": "seen_on", "rule": "min"},
        {"column": "seen_on", "rule": "max", "name": "last_seen_on"},
    ]}  # fmt: skip
    rows = [("v1", "a", None, "w", "x"), ("v1", "a", 7, "w", "x"), ("v1", "a", 3, "w", "x")]
    assert _resolve(identity, VISITS, rows) == [("v1", 3, 7)]


def test_min_by_and_max_by_take_the_value_on_the_ordered_row() -> None:
    identity = {"key_columns": ["visit_id"], "attributes": [
        {"column": "ward", "rule": "min_by", "by": "seen_on", "name": "first_ward"},
        {"column": "ward", "rule": "max_by", "by": "seen_on", "name": "last_ward"},
    ]}  # fmt: skip
    rows = [("v1", "a", 5, "w5", "x"), ("v1", "a", None, "w0", "x"), ("v1", "a", 2, "w2", "x")]
    assert _resolve(identity, VISITS, rows) == [("v1", "w2", "w5")]


def test_a_tie_on_the_ordering_column_resolves_to_the_lowest_value() -> None:
    identity = {"key_columns": ["visit_id"], "attributes": [
        {"column": "ward", "rule": "min_by", "by": "seen_on", "name": "first_ward"},
        {"column": "ward", "rule": "max_by", "by": "seen_on", "name": "last_ward"},
    ]}  # fmt: skip
    rows = [("v1", "a", 1, "wb", "x"), ("v1", "a", 1, "wa", "x"), ("v1", "a", 1, None, "x")]
    for order in (rows, rows[::-1]):
        assert _resolve(identity, VISITS, order) == [("v1", "wa", "wa")]


def test_a_preferred_value_wins_before_the_rule_and_falls_back_without_one() -> None:
    identity = {"key_columns": ["visit_id"], "attributes": [
        {"column": "ward", "rule": "min_by", "by": "seen_on",
         "prefer": {"column": "ward_kind", "value": "primary"}},
        {"column": "seen_on", "rule": "min", "prefer": {"column": "ward_kind", "value": "primary"}},
    ]}  # fmt: skip
    rows = [
        ("v1", "a", 1, "w1", "other"), ("v1", "a", 4, "w4", "primary"),
        ("v1", "a", 3, "w3", "primary"), ("v2", "a", 2, "w2", "other"),
        ("v2", "a", 1, "w1", None),
    ]
    assert _resolve(identity, VISITS, rows) == [("v1", "w3", 3), ("v2", "w1", 1)]


def test_a_key_only_identity_lists_each_entity_once() -> None:
    rows = [("v1", "a", 1, "w", "x"), ("v1", "b", 2, "w", "x"), ("v2", "a", 3, "w", "x")]
    assert _resolve({"key_columns": ["visit_id"]}, VISITS, rows) == [("v1",), ("v2",)]


PUMS_PERSONS = "SERIALNO VARCHAR, SPORDER INTEGER, RELSHIPP VARCHAR, AGEP INTEGER"


def test_pums_person_rows_resolve_to_households() -> None:
    identity = {"key_columns": ["SERIALNO"], "attributes": [
        {"column": "AGEP", "rule": "min_by", "by": "SPORDER", "name": "reference_age",
         "prefer": {"column": "RELSHIPP", "value": "20"}},
        {"column": "AGEP", "rule": "min", "name": "youngest_age"},
        {"column": "AGEP", "rule": "max", "name": "oldest_age"},
    ]}  # fmt: skip
    persons = [
        ("2024HU01", 1, "21", 44), ("2024HU01", 2, "20", 46), ("2024HU01", 3, "25", 9),
        ("2024HU02", 1, "20", 71),
        ("2024GQ03", 1, "37", 19), ("2024GQ03", 2, "38", 22),
    ]
    assert _resolve(identity, PUMS_PERSONS, persons) == [
        ("2024GQ03", 19, 19, 22), ("2024HU01", 46, 9, 46), ("2024HU02", 71, 71, 71),
    ]


PUMS_STATE_PERSONS = "SERIALNO VARCHAR, ST VARCHAR, SPORDER INTEGER, RELSHIPP VARCHAR, AGEP INTEGER"


def test_pums_households_need_a_serial_and_rank_the_reference_person_then_spouse() -> None:
    identity = {"key_columns": ["SERIALNO", "ST"], "exclude_null_keys": ["SERIALNO"],
                "attributes": [
        {"column": "AGEP", "rule": "min_by", "by": "SPORDER", "name": "anchor_age",
         "prefer": {"column": "RELSHIPP", "value": ["20", "21"]}},
    ]}  # fmt: skip
    persons = [
        ("2024HU01", "06", 1, "25", 9), ("2024HU01", "06", 2, "21", 44),
        ("2024HU01", "06", 3, "20", 46),
        ("2024HU02", "06", 1, "25", 12), ("2024HU02", "06", 2, "21", 40),
        ("2024HU03", None, 1, "37", 19), ("2024HU03", None, 2, "38", 22),
        (None, "06", 1, "20", 80),
    ]
    assert _resolve(identity, PUMS_STATE_PERSONS, persons) == [
        ("2024HU01", "06", 46), ("2024HU02", "06", 40), ("2024HU03", None, 19),
    ]


def test_the_cte_names_the_relation_and_the_entity() -> None:
    sql = identity_cte_sql("visit", "cat.sch.visits", EntityIdentity(key_columns=("visit_id",)))
    assert sql.startswith("WITH visit AS (")
    assert "cat.sch.visits" in sql
