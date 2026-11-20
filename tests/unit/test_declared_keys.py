"""A declared foreign key becomes one `JOINS_TO` per constraint, in the C3 shape."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import duckdb
import pytest
from databricks.sql import exc as databricks_exc
from pydantic import SecretStr

from sema.connectors.databricks import DatabricksConnector, TableWorkItem
from sema.connectors.databricks_utils import _build_table_assertion, _make_table_ref
from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.config import DatabricksConfig
from sema.models.extraction import ExtractedForeignKey
from sema.models.physical_key import CanonicalRef, PhysicalKey
from sema.pipeline.normalizer import AssertionNormalizer
from tests.support.connector_fixture import (
    CATALOG,
    COMPAT_SCHEMA,
    HOST,
    MAIN_SCHEMA,
    MAIN_TABLES,
    FakeConnection,
    build_warehouse,
    fixture_environment,
)

pytestmark = pytest.mark.unit

C3_KEYS = {"join_predicates", "from_ref", "to_ref", "constraint_name", "hop_count"}
Predicates = tuple[tuple[str, str, str, str, str], ...]


def _config() -> DatabricksConfig:
    return DatabricksConfig(host=f"https://{HOST}", http_path="/fixture", token=SecretStr("t"))


def _work_item(connector: DatabricksConnector, catalog: str, schema: str, table: str) -> Any:
    return TableWorkItem(catalog, schema, table, _make_table_ref(connector, catalog, schema, table))


def _ref(table: str, catalog: str = CATALOG, schema: str = MAIN_SCHEMA) -> str:
    return CanonicalRef.build("databricks", PhysicalKey(HOST, catalog, schema, table))


def _predicates(assertion: Assertion) -> Predicates:
    return tuple(
        (p["left_table"], p["left_column"], p["right_table"], p["right_column"], p["operator"])
        for p in assertion.payload["join_predicates"]
    )


def _joins(assertions: list[Assertion]) -> list[Assertion]:
    return [a for a in assertions if a.predicate == AssertionPredicate.JOINS_TO]


@pytest.fixture(scope="module")
def fixture_joins() -> Iterator[list[Assertion]]:
    with fixture_environment() as connections:
        connector = DatabricksConnector(config=_config())
        assertions = [
            a for table in MAIN_TABLES
            for a in _build_table_assertion(
                connector, _work_item(connector, CATALOG, MAIN_SCHEMA, table.name),
            )
        ]
        assert not [q for c in connections for q in c.unanswered], "a statement went unanswered"
    yield _joins(assertions)


def _expected() -> set[tuple[str, str, Predicates]]:
    return {
        (_ref(table.name), _ref(key.referenced_table), tuple(
            (key.referenced_table, referenced, table.name, referencing, "=")
            for referencing, referenced in zip(key.columns, key.referenced_columns)
        ))
        for table in MAIN_TABLES for key in table.declared_keys
    }


def test_one_join_per_declared_constraint(fixture_joins: list[Assertion]) -> None:
    found = {(a.subject_ref, a.object_ref, _predicates(a)) for a in fixture_joins}
    assert len(fixture_joins) == len(_expected())
    assert found == _expected()


def test_two_column_key_has_two_predicates_in_ordinal_order(
    fixture_joins: list[Assertion],
) -> None:
    [composite] = [a for a in fixture_joins if len(a.payload["join_predicates"]) == 2]
    assert [p["right_column"] for p in composite.payload["join_predicates"]] == [
        "order_id", "line_no",
    ]


def test_single_column_key_sharing_a_column_is_its_own_assertion(
    fixture_joins: list[Assertion],
) -> None:
    from_shipment = [a for a in fixture_joins if a.subject_ref == _ref("shipment_line")]
    assert sorted(len(a.payload["join_predicates"]) for a in from_shipment) == [1, 2]
    assert {a.object_ref for a in from_shipment} == {_ref("order_line"), _ref("order_header")}


def test_two_roles_between_one_pair_are_two_assertions(fixture_joins: list[Assertion]) -> None:
    roles = [a for a in fixture_joins if a.subject_ref == _ref("transfer")]
    assert sorted(_predicates(a)[0][3] for a in roles) == ["from_account_id", "to_account_id"]
    assert len({a.payload["constraint_name"] for a in roles}) == 2


def test_payload_is_c3_with_the_declared_constraint_names(
    fixture_joins: list[Assertion],
) -> None:
    warehouse = build_warehouse()
    declared = {row[0] for row in warehouse.execute(
        "SELECT constraint_name FROM information_schema.referential_constraints "
        "WHERE constraint_schema = ?", [MAIN_SCHEMA],
    ).fetchall()}
    warehouse.close()
    assert {a.payload["constraint_name"] for a in fixture_joins} == declared
    for join in fixture_joins:
        assert set(join.payload) == C3_KEYS
        assert join.payload["hop_count"] == 1
        assert join.subject_ref == join.payload["to_ref"]
        assert join.object_ref == join.payload["from_ref"]
        assert (join.source, join.confidence) == ("unity_catalog", 0.95)


def test_refs_parse_to_the_fixture_keys(fixture_joins: list[Assertion]) -> None:
    for join in fixture_joins:
        left, _, right, _, _ = _predicates(join)[0]
        assert CanonicalRef.parse(join.payload["from_ref"]) == PhysicalKey(
            HOST, CATALOG, MAIN_SCHEMA, left,
        )
        assert CanonicalRef.parse(join.payload["to_ref"]) == PhysicalKey(
            HOST, CATALOG, MAIN_SCHEMA, right,
        )


_ANSI_VIEWS = {
    "table_constraints": "constraint_catalog VARCHAR, constraint_schema VARCHAR, "
    "constraint_name VARCHAR, table_catalog VARCHAR, table_schema VARCHAR, "
    "table_name VARCHAR, constraint_type VARCHAR",
    "referential_constraints": "constraint_catalog VARCHAR, constraint_schema VARCHAR, "
    "constraint_name VARCHAR, unique_constraint_catalog VARCHAR, "
    "unique_constraint_schema VARCHAR, unique_constraint_name VARCHAR",
    "key_column_usage": "constraint_catalog VARCHAR, constraint_schema VARCHAR, "
    "constraint_name VARCHAR, table_catalog VARCHAR, table_schema VARCHAR, "
    "table_name VARCHAR, column_name VARCHAR, ordinal_position INTEGER, "
    "position_in_unique_constraint INTEGER",
}
_CROSS_ROWS = {
    "table_constraints": [
        ("cat_a", "s_ref", "item_pk", "cat_a", "s_ref", "item", "PRIMARY KEY"),
        ("cat_b", "s_use", "line_item_fk", "cat_b", "s_use", "line", "FOREIGN KEY"),
    ],
    "referential_constraints": [
        ("cat_b", "s_use", "line_item_fk", "cat_a", "s_ref", "item_pk"),
    ],
    "key_column_usage": [
        ("cat_a", "s_ref", "item_pk", "cat_a", "s_ref", "item", "item_no", 1, None),
        ("cat_a", "s_ref", "item_pk", "cat_a", "s_ref", "item", "item_rev", 2, None),
        ("cat_b", "s_use", "line_item_fk", "cat_b", "s_use", "line", "rev", 2, 2),
        ("cat_b", "s_use", "line_item_fk", "cat_b", "s_use", "line", "no", 1, 1),
    ],
}


def _cross_schema_warehouse() -> duckdb.DuckDBPyConnection:
    """DuckDB refuses a cross-catalog key, so the ANSI views are written as rows."""
    con = duckdb.connect(":memory:")
    con.execute(f"CREATE SCHEMA {COMPAT_SCHEMA}")
    for view, columns in _ANSI_VIEWS.items():
        con.execute(f"CREATE TABLE {COMPAT_SCHEMA}.{view} ({columns})")
        for row in _CROSS_ROWS[view]:
            con.execute(
                f"INSERT INTO {COMPAT_SCHEMA}.{view} VALUES ({', '.join('?' * len(row))})", row,
            )
    return con


def test_referenced_relation_catalog_and_schema_are_read_not_assumed() -> None:
    connection = FakeConnection(_cross_schema_warehouse())
    with patch("sema.connectors.databricks.sql_connect", return_value=connection):
        connector = DatabricksConnector(config=_config())
    [join] = _joins(_build_table_assertion(
        connector, _work_item(connector, "cat_b", "s_use", "line"),
    ))
    assert CanonicalRef.parse(join.payload["from_ref"]) == PhysicalKey(
        HOST, "cat_a", "s_ref", "item",
    )
    assert join.subject_ref == _ref("line", "cat_b", "s_use")
    assert _predicates(join) == (
        ("item", "item_no", "line", "no", "="), ("item", "item_rev", "line", "rev", "="),
    )
    assert join.payload["constraint_name"] == "line_item_fk"


@pytest.fixture
def log_lines() -> Iterator[list[str]]:
    from sema.log import logger

    lines: list[str] = []
    handler_id = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
    yield lines
    logger.remove(handler_id)


def test_failed_key_read_is_logged_and_yields_no_keys(log_lines: list[str]) -> None:
    cursor = MagicMock()
    cursor.execute.side_effect = databricks_exc.ServerOperationError("no such view")
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    with patch("sema.connectors.databricks.sql_connect", return_value=connection):
        connector = DatabricksConnector(config=_config())
    assert connector._get_fk_constraints("cat_x", "s_x", "orphan") == []
    assert any("orphan" in line and "no such view" in line for line in log_lines), log_lines


def test_normalizer_emits_the_same_c3_shape() -> None:
    key = ExtractedForeignKey(
        constraint_name="line_item_fk",
        referencing_catalog="cat_b", referencing_schema="s_use", referencing_table="line",
        referenced_catalog="cat_a", referenced_schema="s_ref", referenced_table="item",
        column_pairs=(("no", "item_no"), ("rev", "item_rev")),
    )
    [join] = AssertionNormalizer(workspace=HOST, run_id="r").normalize_foreign_key(key)
    assert join.predicate == AssertionPredicate.JOINS_TO
    assert set(join.payload) == C3_KEYS
    assert (join.subject_ref, join.object_ref) == (
        _ref("line", "cat_b", "s_use"), _ref("item", "cat_a", "s_ref"),
    )
    assert (join.payload["to_ref"], join.payload["from_ref"]) == (
        join.subject_ref, join.object_ref,
    )
    assert _predicates(join) == (
        ("item", "item_no", "line", "no", "="), ("item", "item_rev", "line", "rev", "="),
    )
    assert (join.payload["constraint_name"], join.payload["hop_count"]) == ("line_item_fk", 1)
