"""The entity key: one null-safe surrogate column ``<entity>_key`` per entity row.

A key part may be NULL. The key treats NULL as a value, so it counts the entities a
multi-column ``COUNT(DISTINCT ...)`` drops. Run in DuckDB, no warehouse.
"""

from __future__ import annotations

from typing import Any

import duckdb
import pytest
import sqlglot
import yaml

from sema.compile.entity_identity import identity_cte_sql
from sema.models.target.identity import EntityIdentity

pytestmark = pytest.mark.unit

VISITS = "visit_id VARCHAR, site VARCHAR, seen_on INTEGER, site_label VARCHAR"
VISIT_ROWS = [
    ("v1", "a", 1, "Alpha"), ("v1", "a", 2, "Alpha"),
    ("v1", "b", 3, "Beta"),
    ("v1", None, 4, "Unrecorded site"),
    (None, "a", 5, "Alpha"), (None, "a", 6, "Alpha"),
]  # fmt: skip
VISIT_KEYS = {"key_columns": ["visit_id", "site"]}
NULL_AS_VALUE_ENTITIES = 4
NON_NULL_ENTITIES = 2


def _warehouse(columns: str, rows: list[tuple[Any, ...]]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE TABLE src ({columns})")
    marks = ", ".join("?" for _ in columns.split(","))
    con.executemany(f"INSERT INTO src VALUES ({marks})", rows)
    return con


def _query(
    con: duckdb.DuckDBPyConnection, entity: str, identity: dict[str, Any], select: str
) -> list[tuple[Any, ...]]:
    cte = identity_cte_sql(entity, "src", EntityIdentity.model_validate(identity), "duckdb")
    return con.execute(f"{cte} {select}").fetchall()


def _distinct_keys(con: duckdb.DuckDBPyConnection, entity: str, identity: dict[str, Any]) -> int:
    ((count,),) = _query(con, entity, identity,
                         f"SELECT COUNT(DISTINCT {entity}_key) FROM {entity}")  # fmt: skip
    return int(count)


def _multi_column_distinct(con: duckdb.DuckDBPyConnection, keys: list[str]) -> int:
    """``COUNT(DISTINCT a, b)``: a row with any NULL key part is not counted."""
    listed = ", ".join(f'"{k}"' for k in keys)
    present = " AND ".join(f'"{k}" IS NOT NULL' for k in keys)
    ((count,),) = con.execute(
        f"SELECT count(*) FROM (SELECT DISTINCT {listed} FROM src WHERE {present})"
    ).fetchall()
    return int(count)


def _null_as_value_distinct(con: duckdb.DuckDBPyConnection, keys: list[str]) -> int:
    listed = ", ".join(f'"{k}"' for k in keys)
    query = f"SELECT count(*) FROM (SELECT DISTINCT {listed} FROM src)"
    ((count,),) = con.execute(query).fetchall()
    return int(count)


def test_the_entity_key_counts_a_null_key_part_as_a_value() -> None:
    con = _warehouse(VISITS, VISIT_ROWS)
    keys = VISIT_KEYS["key_columns"]
    assert _null_as_value_distinct(con, keys) == NULL_AS_VALUE_ENTITIES
    assert _multi_column_distinct(con, keys) == NON_NULL_ENTITIES
    assert _distinct_keys(con, "visit", VISIT_KEYS) == NULL_AS_VALUE_ENTITIES


def test_every_entity_row_has_one_distinct_non_null_key() -> None:
    con = _warehouse(VISITS, VISIT_ROWS)
    select = "SELECT count(*), count(visit_key), COUNT(DISTINCT visit_key) FROM visit"
    assert _query(con, "visit", VISIT_KEYS, select) == [(4, 4, 4)]


def test_each_null_key_row_has_its_own_key_when_null_keys_do_not_group() -> None:
    con = _warehouse(VISITS, VISIT_ROWS)
    identity = {**VISIT_KEYS, "null_keys_group": False}
    assert _distinct_keys(con, "visit", identity) == 5


def test_the_entity_key_of_a_key_value_does_not_depend_on_row_order() -> None:
    select = "SELECT visit_id, site, visit_key FROM visit ORDER BY ALL NULLS LAST"
    forward = _query(_warehouse(VISITS, VISIT_ROWS), "visit", VISIT_KEYS, select)
    backward = _query(_warehouse(VISITS, VISIT_ROWS[::-1]), "visit", VISIT_KEYS, select)
    assert forward == backward


def test_a_companion_label_is_null_when_its_code_is_null() -> None:
    identity = {**VISIT_KEYS, "attributes": [
        {"column": "site_label", "rule": "min_by", "by": "seen_on", "label_of": "site"},
    ]}  # fmt: skip
    select = "SELECT visit_id, site, site_label FROM visit ORDER BY ALL NULLS LAST"
    assert _query(_warehouse(VISITS, VISIT_ROWS), "visit", identity, select) == [
        ("v1", "a", "Alpha"), ("v1", "b", "Beta"), ("v1", None, None), (None, "a", "Alpha"),
    ]


@pytest.mark.parametrize("dialect", ["databricks", "spark", "postgres"])
def test_each_warehouse_dialect_projects_the_entity_key(dialect: str) -> None:
    identity = EntityIdentity.model_validate(VISIT_KEYS)
    cte = identity_cte_sql("visit", "cat.sch.visits", identity, dialect=dialect)
    tree = sqlglot.parse_one(f"{cte} SELECT visit_key FROM visit", read=dialect)
    (body,) = [c.this for c in tree.find_all(sqlglot.exp.CTE)]
    assert "visit_key" in body.named_selects


ACRIS_LEGALS = '"DOCUMENT ID" VARCHAR, "GOOD THROUGH DATE" VARCHAR, BOROUGH VARCHAR, LOT VARCHAR'
ACRIS_DOCUMENT = """
key_columns: [DOCUMENT ID, GOOD THROUGH DATE]
attributes:
  - {column: LOT, rule: min, name: first_lot}
"""


def test_an_acris_document_entity_keys_documents_with_an_unknown_good_through_date() -> None:
    lots = [
        ("2024010100001", "2025-06-30", "1", "0010"), ("2024010100001", "2025-06-30", "1", "0011"),
        ("2024010100002", "2025-06-30", "3", "0007"),
        ("2024010100003", None, "4", "0002"), ("2024010100003", None, "4", "0003"),
    ]  # fmt: skip
    con = _warehouse(ACRIS_LEGALS, lots)
    identity = yaml.safe_load(ACRIS_DOCUMENT)
    assert _multi_column_distinct(con, identity["key_columns"]) == 2
    assert _distinct_keys(con, "document", identity) == 3


PUMS_PERSONS = "SERIALNO VARCHAR, SPORDER INTEGER, RELSHIPP VARCHAR, AGEP INTEGER"
PUMS_PERSON = """
key_columns: [SERIALNO, SPORDER]
attributes:
  - {column: AGEP, rule: max, name: age}
"""


def test_a_pums_person_entity_keys_persons_whose_order_is_unrecorded() -> None:
    persons = [
        ("2024HU01", 1, "20", 46), ("2024HU01", 1, "20", 46), ("2024HU01", 2, "21", 44),
        ("2024GQ03", None, "37", 19), ("2024GQ04", None, "37", 22),
    ]  # fmt: skip
    con = _warehouse(PUMS_PERSONS, persons)
    identity = yaml.safe_load(PUMS_PERSON)
    assert _multi_column_distinct(con, identity["key_columns"]) == 2
    assert _distinct_keys(con, "person", identity) == 4


@pytest.mark.parametrize("label_of", ["site_label", "seen_on"])
def test_a_companion_label_names_another_resolved_column(label_of: str) -> None:
    identity = {**VISIT_KEYS, "attributes": [
        {"column": "site_label", "rule": "min_by", "by": "seen_on", "label_of": label_of},
    ]}  # fmt: skip
    with pytest.raises(ValueError, match="not another resolved column"):
        EntityIdentity.model_validate(identity)


@pytest.mark.parametrize(("identity", "clash"), [
    ({"key_columns": ["e_key"]}, "e_key"),
    ({"key_columns": ["a"], "attributes": [{"column": "a", "rule": "min", "name": "E_Key"}]},
     "E_Key"),
])  # fmt: skip
def test_an_entity_key_equal_to_a_projected_column_is_refused(
    identity: dict[str, Any], clash: str
) -> None:
    """Two ``e_key`` columns would let ``COUNT(DISTINCT e_key)`` count the source column."""
    with pytest.raises(ValueError, match=f"e_key.*{clash}"):
        identity_cte_sql("e", "src", EntityIdentity.model_validate(identity), "duckdb")
