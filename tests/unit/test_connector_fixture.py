"""The neutral connector fixture answers the connector honestly and holds the join shapes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import duckdb
import pytest
from pydantic import BaseModel, SecretStr

from sema.connectors.databricks import DatabricksConnector
from sema.llm_client import SynonymExpansion, TableSummary, VocabularyDetection
from sema.models.assertions import AssertionPredicate
from sema.models.config import Neo4jConfig
from sema.models.physical_key import CanonicalRef
from sema.models.stages import (
    StageAResult,
    StageBBatchResult,
    StageCBatchResult,
    StageCResult,
)
from tests.support.connector_fixture import (
    CATALOG,
    DECOY_SCHEMA,
    DECOY_TABLES,
    EXPECTED_DETECTED_JOINS,
    MAIN_SCHEMA,
    MAIN_TABLES,
    SCHEMA_TABLES,
    FakeConnection,
    FixtureJoin,
    build_warehouse,
    fixture_build_config,
    fixture_environment,
    make_fake_llm_client,
)

pytestmark = pytest.mark.unit

DECLARED_KEYS_SQL = (
    "SELECT kcu.table_schema, kcu.table_name, kcu.constraint_name, kcu.column_name, "
    "kcu.ordinal_position, ref.table_name, ref.column_name "
    "FROM `system`.`information_schema`.`referential_constraints` rc "
    "JOIN `system`.`information_schema`.`key_column_usage` kcu "
    "ON kcu.constraint_catalog = rc.constraint_catalog "
    "AND kcu.constraint_schema = rc.constraint_schema "
    "AND kcu.constraint_name = rc.constraint_name "
    "JOIN `system`.`information_schema`.`key_column_usage` ref "
    "ON ref.constraint_catalog = rc.unique_constraint_catalog "
    "AND ref.constraint_schema = rc.unique_constraint_schema "
    "AND ref.constraint_name = rc.unique_constraint_name "
    "AND ref.ordinal_position = kcu.position_in_unique_constraint "
    f"WHERE kcu.table_catalog = '{CATALOG}' "
    "ORDER BY kcu.table_schema, kcu.table_name, kcu.constraint_name, kcu.ordinal_position"
)


@pytest.fixture
def warehouse() -> Iterator[duckdb.DuckDBPyConnection]:
    con = build_warehouse()
    yield con
    con.close()


def _query(connection: FakeConnection, sql: str) -> list[tuple[Any, ...]]:
    with connection.cursor() as cursor:
        cursor.execute(sql)
        return cursor.fetchall()


def test_information_schema_exposes_every_declared_key_with_positions(
    warehouse: duckdb.DuckDBPyConnection,
) -> None:
    rows = _query(FakeConnection(warehouse), DECLARED_KEYS_SQL)
    found: dict[tuple[str, str, str], list[tuple[int, str, str, str]]] = {}
    for schema, table, constraint, column, position, ref_table, ref_column in rows:
        found.setdefault((schema, table, constraint), []).append(
            (position, column, ref_table, ref_column),
        )
    observed = {
        (schema, table, tuple((c, rt, rc) for _, c, rt, rc in sorted(pairs)))
        for (schema, table, _), pairs in found.items()
    }
    expected = {
        (schema, table.name, tuple(
            (column, key.referenced_table, ref_column)
            for column, ref_column in zip(key.columns, key.referenced_columns)
        ))
        for schema, tables in SCHEMA_TABLES.items()
        for table in tables
        for key in table.declared_keys
    }
    assert observed == expected
    assert len(found) == sum(
        len(table.declared_keys) for tables in SCHEMA_TABLES.values() for table in tables
    )


def test_connector_reads_every_table_and_column_of_both_schemas() -> None:
    expected = {
        (schema, table.name, column)
        for schema, tables in SCHEMA_TABLES.items()
        for table in tables
        for column in table.column_names
    }
    neo4j = Neo4jConfig(uri="bolt://unused", user="unused", password=SecretStr("unused"))
    config = fixture_build_config(neo4j)
    with fixture_environment():
        connector = DatabricksConnector(config.databricks, config.profiling)
        work_items = connector.discover_tables(CATALOG, config.schemas)
        assertions = [a for item in work_items for a in connector.extract_table(item)]
    keys = [
        CanonicalRef.parse(a.subject_ref)
        for a in assertions if a.predicate == AssertionPredicate.COLUMN_EXISTS
    ]
    assert {(k.schema, k.table, k.column) for k in keys} == expected
    assert {(item.schema, item.table_name) for item in work_items} == {
        (schema, name) for schema, name, _ in expected
    }


def test_unanswerable_statement_is_recorded_and_raised(
    warehouse: duckdb.DuckDBPyConnection,
) -> None:
    connection = FakeConnection(warehouse)
    statement = f"SELECT no_such_column FROM `{CATALOG}`.`{MAIN_SCHEMA}`.`customer`"
    with pytest.raises(duckdb.Error):
        _query(connection, statement)
    assert len(connection.unanswered) == 1
    assert statement in connection.unanswered[0]


def test_decoy_schema_mirrors_a_joined_main_table() -> None:
    main = {table.name: table.columns for table in MAIN_TABLES}
    referenced = {key.referenced_table for t in MAIN_TABLES for key in t.declared_keys}
    for decoy in DECOY_TABLES:
        assert main[decoy.name] == decoy.columns
        assert decoy.name in referenced
    assert DECOY_SCHEMA != MAIN_SCHEMA


def _is_unique(con: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    distinct, total = con.execute(
        f"SELECT COUNT(DISTINCT {column}), COUNT(*) FROM {CATALOG}.{MAIN_SCHEMA}.{table}"
    ).fetchone() or (0, 0)
    return bool(total) and distinct == total


def _uncontained(con: duckdb.DuckDBPyConnection, join: FixtureJoin) -> int:
    row = con.execute(
        f"SELECT COUNT(*) FROM (SELECT DISTINCT {join.referencing_column} AS v "
        f"FROM {CATALOG}.{MAIN_SCHEMA}.{join.referencing_table}) "
        f"WHERE v NOT IN (SELECT {join.referenced_column} "
        f"FROM {CATALOG}.{MAIN_SCHEMA}.{join.referenced_table})"
    ).fetchone()
    return int(row[0]) if row else -1


def test_every_expected_join_holds_in_the_data(warehouse: duckdb.DuckDBPyConnection) -> None:
    for join in EXPECTED_DETECTED_JOINS:
        assert join.referenced_table and join.referenced_column
        assert _is_unique(warehouse, join.referenced_table, join.referenced_column), join
        assert _uncontained(warehouse, join) == 0, join


def test_each_decoy_tempts_one_predicate(warehouse: duckdb.DuckDBPyConnection) -> None:
    coincidence = FixtureJoin("order_line", "qty", "supplier", "supplier_id")
    assert _is_unique(warehouse, "supplier", "supplier_id")
    assert _uncontained(warehouse, coincidence) == 0
    affinity = FixtureJoin("order_header", "legacy_customer_id", "customer", "customer_id")
    assert _uncontained(warehouse, affinity) > 0
    non_unique = FixtureJoin("shipment_line", "order_id", "order_line", "order_id")
    assert _uncontained(warehouse, non_unique) == 0
    assert not _is_unique(warehouse, "order_line", "order_id")


@pytest.mark.parametrize(
    "schema",
    [StageAResult, StageBBatchResult, StageCBatchResult, StageCResult,
     TableSummary, VocabularyDetection, SynonymExpansion],
)
def test_fake_model_answers_every_stage_schema(schema: type[BaseModel]) -> None:
    table_ref = f"databricks://host/{CATALOG}/{MAIN_SCHEMA}/order_line"
    answer = make_fake_llm_client().invoke("order_id line_no qty", schema, table_ref=table_ref)
    assert isinstance(answer, schema)


def test_fake_model_names_properties_for_the_columns_in_the_prompt() -> None:
    table_ref = f"databricks://host/{CATALOG}/{MAIN_SCHEMA}/order_line"
    answer = make_fake_llm_client().invoke("order_id qty", StageBBatchResult, table_ref=table_ref)
    assert {c.column for c in answer.columns} == {"order_id", "qty"}
