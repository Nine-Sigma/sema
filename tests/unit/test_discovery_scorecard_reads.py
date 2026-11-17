"""The discovery scorecard's data reads: rechecks, declared keys and oracle files.

Every read runs through the DuckDB connector on a database file built in `tmp_path`: the
neutral connector fixture, or a three-table schema declaring a one-to-one and an ordinary
foreign key.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from sema.connectors.duckdb import DuckDBConnector
from sema.eval.discovery_scorecard import declared_constraints, recheck
from sema.eval.discovery_scorecard_utils import read_oracle, unique_keys
from sema.models.config import DuckDBConfig
from tests.support.connector_fixture import (
    CATALOG,
    DECOY_JOINS,
    MAIN_SCHEMA,
    MAIN_TABLES,
    DeclaredKey,
    FixtureTable,
    write_fixture_database,
)
from tests.support.connector_fixture_scorecard import (
    DECLARED_JOIN_KEYS,
    DETECTED_JOIN_KEYS,
    Key,
    join_key,
)

pytestmark = pytest.mark.unit

PRIMARY_KEYS = {table.name: table.primary_key for table in MAIN_TABLES}
DECOYS_ON_NON_UNIQUE_COLUMNS = frozenset(
    join_key(j.referencing_table, [j.referencing_column], j.referenced_table,
             [j.referenced_column])
    for j in DECOY_JOINS
    if j.referenced_table and j.referenced_column
    and PRIMARY_KEYS[j.referenced_table] != (j.referenced_column,)
)
SINGLE_COLUMN_JOINS = frozenset(
    key for key in DECLARED_JOIN_KEYS | DETECTED_JOIN_KEYS if "+" not in key[1]
)

SCHEMA, KEYS_CATALOG = "s", "keys"
PARENT = FixtureTable("parent", (("parent_id", "INTEGER"),), ("parent_id",),
                      "SELECT i FROM range(1, 11) t(i)")
EXTENSION = FixtureTable(
    "extension", (("parent_id", "INTEGER"), ("note", "VARCHAR")), ("parent_id",),
    "SELECT i, 'n' FROM range(1, 6) t(i)",
    (DeclaredKey(("parent_id",), "parent", ("parent_id",)),),
)
CHILD = FixtureTable(
    "child", (("child_id", "INTEGER"), ("parent_id", "INTEGER")), ("child_id",),
    "SELECT i, 1 + (i % 10) FROM range(1, 31) t(i)",
    (DeclaredKey(("parent_id",), "parent", ("parent_id",)),),
)


def _connector(path: Path) -> Iterator[DuckDBConnector]:
    opened = DuckDBConnector(DuckDBConfig(path=str(path)))
    yield opened
    opened.close()


@pytest.fixture
def fixture_connector(tmp_path: Path) -> Iterator[DuckDBConnector]:
    yield from _connector(write_fixture_database(tmp_path))


@pytest.fixture
def keys_connector(tmp_path: Path) -> Iterator[DuckDBConnector]:
    tables = {SCHEMA: (PARENT, EXTENSION, CHILD)}
    yield from _connector(write_fixture_database(tmp_path, tables, catalog=KEYS_CATALOG))


def _rechecked(connector: DuckDBConnector, keys: frozenset[Key]) -> int:
    produced = {"join": dict.fromkeys(keys, "key_only")}
    return recheck(connector, CATALOG, [MAIN_SCHEMA], produced)["join"]


def test_the_fixture_reading_behind_the_recheck_cases() -> None:
    assert len(DECOYS_ON_NON_UNIQUE_COLUMNS) == 2
    assert len(SINGLE_COLUMN_JOINS) == 11


@pytest.mark.parametrize("key", sorted(DECOYS_ON_NON_UNIQUE_COLUMNS))
def test_a_decoy_join_onto_a_non_unique_column_does_not_recheck(
    fixture_connector: DuckDBConnector, key: Key,
) -> None:
    assert _rechecked(fixture_connector, frozenset({key})) == 0


def test_every_single_column_declared_or_detected_join_rechecks(
    fixture_connector: DuckDBConnector,
) -> None:
    assert _rechecked(fixture_connector, SINGLE_COLUMN_JOINS) == len(SINGLE_COLUMN_JOINS)


def test_a_join_whose_values_are_not_contained_does_not_recheck(
    fixture_connector: DuckDBConnector,
) -> None:
    uncontained = join_key("order_header", ["legacy_customer_id"], "customer", ["customer_id"])

    assert _rechecked(fixture_connector, frozenset({uncontained})) == 0


def test_unique_keys_groups_each_constraints_columns() -> None:
    rows = [("T", "pk_t", "b"), ("T", "pk_t", "a"), ("T", "uq_t", "c")]

    assert unique_keys(rows) == {("t", "a+b"), ("t", "c")}


def test_a_foreign_key_on_declared_unique_columns_is_one_to_one(
    keys_connector: DuckDBConnector,
) -> None:
    oracle = declared_constraints(keys_connector, KEYS_CATALOG, [SCHEMA])

    assert oracle["join"] == {
        ("extension", "parent_id", "parent", "parent_id"): "one_to_one",
        ("child", "parent_id", "parent", "parent_id"): "many_to_one",
    }
    assert oracle["candidate_key"] == {
        ("parent", "parent_id"): "unique", ("extension", "parent_id"): "unique",
        ("child", "child_id"): "unique",
    }


def _claim(kind: str, key: list[str], value: object) -> str:
    return json.dumps({"kind": kind, "key": key, "value": value, "basis": "data", "cite": "c"})


def test_read_oracle_groups_claims_by_kind_and_keys_them_as_tuples(tmp_path: Path) -> None:
    path = tmp_path / "oracle.jsonl"
    path.write_text("\n".join([
        _claim("join", ["a", "x", "b", "y"], "many_to_one"), "",
        _claim("candidate_key", ["b", "y"], "unique"),
    ]))

    assert read_oracle(path, ["join", "candidate_key"]) == {
        "join": {("a", "x", "b", "y"): "many_to_one"},
        "candidate_key": {("b", "y"): "unique"},
    }


def test_read_oracle_refuses_an_unknown_kind_with_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "oracle.jsonl"
    path.write_text("\n".join([
        _claim("join", ["a", "x", "b", "y"], "many_to_one"), _claim("guess", ["a"], 1),
    ]))

    with pytest.raises(ValueError, match=rf"{path.name}:2: unknown fact kind 'guess'"):
        read_oracle(path, ["join"])
