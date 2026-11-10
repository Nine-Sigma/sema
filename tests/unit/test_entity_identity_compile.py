"""Entity identity compiles to one executable CTE (run in DuckDB, no warehouse)."""

from __future__ import annotations

from typing import Any

import duckdb
import pytest
import sqlglot
import sqlglot.expressions as exp

from sema.compile.entity_identity import identity_cte_sql
from sema.models.target.identity import EntityIdentity

pytestmark = pytest.mark.unit

VISITS = "visit_id VARCHAR, site VARCHAR, seen_on INTEGER, ward VARCHAR, ward_kind VARCHAR"


def _resolve(
    identity: dict[str, Any], columns: str, rows: list[tuple[Any, ...]], dialect: str | None = None
) -> list[Any]:
    con = duckdb.connect()
    con.execute(f"CREATE TABLE src ({columns})")
    marks = ", ".join("?" for _ in columns.split(","))
    con.executemany(f"INSERT INTO src VALUES ({marks})", rows)
    cte = identity_cte_sql("entity", "src", EntityIdentity.model_validate(identity), dialect)
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


WAREHOUSE_DIALECTS = ("databricks", "spark", "postgres")


@pytest.mark.parametrize("dialect", ["duckdb", *WAREHOUSE_DIALECTS])
def test_each_row_number_window_is_ordered(dialect: str) -> None:
    identity = EntityIdentity.model_validate({"key_columns": ["visit_id", "site"],
        "null_keys_group": False, "attributes": FIRST_SEEN})  # fmt: skip
    sql = identity_cte_sql("entity", "cat.sch.visits", identity, dialect=dialect)
    tree = sqlglot.parse_one(f"{sql} SELECT * FROM entity", read=dialect)
    windows = list(tree.find_all(exp.Window))
    assert len(windows) == 2
    assert all(w.args.get("order") is not None for w in windows)


RESERVED_AND_UPPER = {"key_columns": ["order", "SERIALNO"], "null_keys_group": False,
                      "attributes": [{"column": "select", "rule": "min"}]}  # fmt: skip
RESERVED_COLUMNS = '"order" INTEGER, SERIALNO VARCHAR, "select" INTEGER'


@pytest.mark.parametrize("dialect", ["duckdb", *WAREHOUSE_DIALECTS])
def test_every_column_is_quoted_in_the_warehouse_dialect(dialect: str) -> None:
    identity = EntityIdentity.model_validate(RESERVED_AND_UPPER)
    sql = identity_cte_sql("entity", "cat.sch.visits", identity, dialect=dialect)
    tree = sqlglot.parse_one(f"{sql} SELECT * FROM entity", read=dialect)
    columns = [c.this for c in tree.find_all(exp.Column)]
    assert {c.name for c in columns} >= {"order", "SERIALNO", "select"}
    assert all(c.quoted for c in columns)


@pytest.mark.parametrize("dialect", ["duckdb", "postgres"])
def test_reserved_and_upper_case_columns_execute(dialect: str) -> None:
    rows = [(1, "2024HU01", 3), (1, "2024HU01", 2), (None, "2024HU02", 5), (None, "2024HU02", 4)]
    assert _resolve(RESERVED_AND_UPPER, RESERVED_COLUMNS, rows, dialect) == [
        (1, "2024HU01", 2), (None, "2024HU02", 4), (None, "2024HU02", 5),
    ]


@pytest.mark.parametrize(("rule", "by_rule"), [("min", "w1"), ("min_by", "w2")])
def test_a_preferred_row_without_a_value_yields_to_the_rule(rule: str, by_rule: str) -> None:
    by = {"by": "seen_on"} if rule == "min_by" else {}
    prefer = {"column": "ward_kind", "value": "primary"}
    attribute = {"column": "ward", "rule": rule, **by, "prefer": prefer}
    identity = {"key_columns": ["visit_id"], "attributes": [attribute]}
    rows = [("v1", "a", 3, None, "primary"), ("v1", "a", 2, "w2", "other"),
            ("v1", "a", 4, "w1", "other"), ("v2", "a", 1, None, "primary")]  # fmt: skip
    assert _resolve(identity, VISITS, rows) == [("v1", by_rule), ("v2", None)]


def test_source_columns_named_like_internal_columns_do_not_change_the_result() -> None:
    identity = {"key_columns": ["k"], "null_keys_group": False, "attributes": [
        {"column": "v", "rule": "min"}, {"column": "_rank_v", "rule": "max", "name": "r"},
    ]}  # fmt: skip
    columns = "k VARCHAR, v INTEGER, _rank_v INTEGER, _null_key_row INTEGER"
    rows = [("a", 2, 7, 9), ("a", 1, 8, 9), (None, 3, 9, 1), (None, 4, 9, 1)]
    assert _resolve(identity, columns, rows) == [("a", 1, 8), (None, 3, 9), (None, 4, 9)]


ONE_PART_NULLABLE = {"key_columns": ["a_part", "b_part"], "exclude_null_keys": ["a_part"]}
NO_PART_NULLABLE = {"key_columns": ["household_serial", "member_ordinal"],
                    "exclude_null_keys": True}  # fmt: skip


def _lint(sql: str, identity: dict[str, Any]) -> list[Any]:
    from sema.compile.entity_identity import composite_key_count_findings

    statement = sqlglot.parse_one(sql)
    return list(composite_key_count_findings(statement, EntityIdentity.model_validate(identity)))


def test_multi_column_count_distinct_over_a_key_with_a_nullable_part_is_flagged() -> None:
    [finding] = _lint("SELECT COUNT(DISTINCT a_part, b_part) FROM src", ONE_PART_NULLABLE)
    assert finding.key_columns == ("a_part", "b_part")
    assert finding.columns == ("a_part", "b_part")
    assert finding.nullable_columns == ("b_part",)
    assert "b_part" in finding.message


def test_single_column_count_distinct_is_not_flagged() -> None:
    assert not _lint("SELECT COUNT(DISTINCT b_part) FROM src", ONE_PART_NULLABLE)


def test_composite_key_with_no_nullable_part_is_not_flagged() -> None:
    sql = "SELECT COUNT(DISTINCT household_serial, member_ordinal) FROM src"
    assert not _lint(sql, NO_PART_NULLABLE)


def test_qualified_reordered_key_count_inside_a_cte_is_flagged() -> None:
    sql = ("WITH k AS (SELECT a_part, b_part FROM src), "
           "n AS (SELECT COUNT(DISTINCT k.b_part, k.a_part) AS entities FROM k) "
           "SELECT entities FROM n")  # fmt: skip
    [finding] = _lint(sql, ONE_PART_NULLABLE)
    assert finding.columns == ("b_part", "a_part")
    assert finding.nullable_columns == ("b_part",)


def test_count_with_a_nullable_key_part_and_a_non_key_column_is_flagged() -> None:
    [finding] = _lint("SELECT COUNT(DISTINCT b_part, visit_day) FROM src", ONE_PART_NULLABLE)
    assert finding.columns == ("b_part", "visit_day")
    assert finding.nullable_columns == ("b_part",)


def test_key_names_match_counted_columns_in_another_case_in_the_warehouse_dialect() -> None:
    from sema.compile.entity_identity import composite_key_count_findings

    identity = EntityIdentity(key_columns=("SERIALNO", "SPORDER"),
                              exclude_null_keys=("SERIALNO",))  # fmt: skip
    sql = "SELECT COUNT(DISTINCT serialno, sporder) FROM src"
    statement = sqlglot.parse_one(sql, read="databricks")
    [finding] = composite_key_count_findings(statement, identity)
    assert finding.columns == ("serialno", "sporder")
    assert finding.key_columns == ("SERIALNO", "SPORDER")
    assert finding.nullable_columns == ("SPORDER",)


def test_upper_case_counted_columns_match_a_lower_case_key() -> None:
    [finding] = _lint("SELECT COUNT(DISTINCT A_PART, B_PART) FROM src", ONE_PART_NULLABLE)
    assert finding.columns == ("A_PART", "B_PART")
    assert finding.nullable_columns == ("b_part",)
