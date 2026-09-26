"""DatabricksConnector.read_schema_metadata: three information_schema reads per
schema, typed Unreadable on warehouse errors (design 6.1.1, task LN3)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
import sqlglot
from databricks.sql import exc as databricks_exc
from sqlglot import exp

from sema.connectors.databricks import DatabricksConnector
from sema.models.config import DatabricksConfig, ProfilingConfig

pytestmark = pytest.mark.unit

CATALOG = "cat_a"
SCHEMA = "sch_a"
VIEWS = ("tables", "columns", "views")
VIEW_SQL = "SELECT o.id, o.amount FROM cat_a.sch_a.orders AS o WHERE o.id IS NOT NULL"

OBJECT_ROWS = [
    {"table_name": "orders", "table_type": "MANAGED"},
    {"table_name": "orders_v", "table_type": "VIEW"},
    {"table_name": "Mixed_Case", "table_type": "EXTERNAL"},
    {"table_name": "hidden_v", "table_type": "VIEW"},
]


def _col(table: str, name: str, full_type: str, comment: str | None, pos: int) -> dict[str, Any]:
    return {
        "table_name": table,
        "column_name": name,
        "full_data_type": full_type,
        "data_type": full_type.split("(")[0],
        "comment": comment,
        "ordinal_position": pos,
    }


COLUMN_ROWS = [
    _col("Mixed_Case", "code", "STRING", "a code", 0),
    _col("hidden_v", "id", "BIGINT", None, 0),
    _col("orders", "id", "BIGINT", "row id", 0),
    _col("orders", "amount", "DECIMAL(10,2)", "", 1),
    _col("orders", "status", "STRING", None, 2),
    _col("orders_v", "id", "BIGINT", None, 0),
    _col("orders_v", "amount", "DECIMAL(10,2)", "order amount", 1),
]

VIEW_ROWS = [
    {"table_name": "orders_v", "view_definition": VIEW_SQL},
    {"table_name": "hidden_v", "view_definition": None},
]


def _parse(sql: str) -> exp.Expression:
    return sqlglot.parse_one(sql, read="databricks")


def _target_view(sql: str) -> str | None:
    normalized = sql.replace("`", "").lower()
    hits = [v for v in VIEWS if f"information_schema.{v}" in normalized]
    return hits[0] if len(hits) == 1 else None


def _where_equalities(tree: exp.Expression, params: dict[str, Any]) -> dict[str, list[Any]]:
    found: dict[str, list[Any]] = {}
    for eq in tree.find_all(exp.EQ):
        column, value = eq.this, eq.expression
        if isinstance(value, exp.Column):
            column, value = value, column
        if not isinstance(column, exp.Column):
            continue
        if isinstance(value, exp.Placeholder):
            resolved: Any = params[value.name]
        elif isinstance(value, exp.Literal):
            resolved = value.this
        else:
            continue
        found.setdefault(column.name.lower(), []).append(resolved)
    return found


class FakeWarehouse:
    """Records every statement; answers the three information_schema reads by
    the column names each SELECT projects, so the test does not fix column order."""

    def __init__(
        self,
        fail_on: str | None = None,
        error: Exception | None = None,
        raw_rows: dict[str, list[tuple[Any, ...]]] | None = None,
    ) -> None:
        self.rows = {"tables": OBJECT_ROWS, "columns": COLUMN_ROWS, "views": VIEW_ROWS}
        self.fail_on = fail_on
        self.error = error
        self.raw_rows = raw_rows or {}
        self.statements: list[tuple[str, dict[str, Any]]] = []

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self) -> None:
        return None


class FakeCursor:
    def __init__(self, warehouse: FakeWarehouse) -> None:
        self._warehouse = warehouse
        self._result: list[tuple[Any, ...]] = []
        self.description: list[tuple[Any, ...]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def close(self) -> None:
        return None

    def execute(self, operation: str, parameters: Any = None, **kwargs: Any) -> None:
        params = dict(parameters) if isinstance(parameters, dict) else {}
        self._warehouse.statements.append((operation, params))
        view = _target_view(operation)
        if view is not None and view == self._warehouse.fail_on:
            assert self._warehouse.error is not None
            raise self._warehouse.error
        if view in self._warehouse.raw_rows:
            self._result = self._warehouse.raw_rows[view]
            return
        names = [p.alias_or_name.lower() for p in _parse(operation).expressions]
        self.description = [(n,) for n in names]
        rows = self._warehouse.rows.get(view or "", [])
        self._result = [tuple(row[n] for n in names) for row in rows]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._result


def _connector(warehouse: FakeWarehouse) -> DatabricksConnector:
    with patch("sema.connectors.databricks.sql_connect", return_value=warehouse):
        config = DatabricksConfig(
            host="https://test.databricks.com",
            token="dapi123",
            http_path="/sql/1.0/warehouses/test",
        )
        conn = DatabricksConnector(config=config, profiling=ProfilingConfig())
    conn._connection = warehouse
    return conn


@pytest.fixture
def log_lines() -> Iterator[list[str]]:
    from sema.log import logger

    lines: list[str] = []
    handler_id = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
    yield lines
    logger.remove(handler_id)


def _read(warehouse: FakeWarehouse, catalog: str = CATALOG, schema: str = SCHEMA) -> Any:
    return _connector(warehouse).read_schema_metadata(catalog, schema)


# --- acceptance 1: exactly three metadata statements, filtered, no data read ---


def test_issues_exactly_one_statement_per_information_schema_view() -> None:
    warehouse = FakeWarehouse()

    _read(warehouse)

    assert sorted(_target_view(sql) or "other" for sql, _ in warehouse.statements) == sorted(VIEWS)


def test_every_statement_is_filtered_by_catalog_and_schema() -> None:
    warehouse = FakeWarehouse()

    _read(warehouse)

    for sql, params in warehouse.statements:
        equalities = _where_equalities(_parse(sql), params)
        assert equalities.get("table_catalog") == [CATALOG], sql
        assert equalities.get("table_schema") == [SCHEMA], sql


def test_no_statement_reads_a_data_table_or_samples() -> None:
    warehouse = FakeWarehouse()

    _read(warehouse)

    for sql, _ in warehouse.statements:
        tree = _parse(sql)
        assert isinstance(tree, exp.Select), sql
        assert tree.args.get("limit") is None, sql
        assert tree.find(exp.TableSample) is None, sql
        for table in tree.find_all(exp.Table):
            assert table.db.lower() == "information_schema", sql


def test_columns_are_read_in_table_then_ordinal_order() -> None:
    warehouse = FakeWarehouse()

    _read(warehouse)

    (columns_sql,) = [sql for sql, _ in warehouse.statements if _target_view(sql) == "columns"]
    order = _parse(columns_sql).args.get("order")
    assert order is not None
    assert [o.this.name.lower() for o in order.expressions] == ["table_name", "ordinal_position"]


# --- acceptance 2: rows map to SchemaMetadata ---


def test_rows_map_to_schema_metadata_objects_and_types() -> None:
    from sema.models.schema_metadata import SchemaMetadata

    result = _read(FakeWarehouse())

    assert isinstance(result, SchemaMetadata)
    assert (result.catalog, result.schema_name) == (CATALOG, SCHEMA)
    types = {o.name: o.object_type for o in result.objects}
    assert types == {"orders": "MANAGED", "orders_v": "VIEW", "Mixed_Case": "EXTERNAL", "hidden_v": "VIEW"}
    assert all((o.catalog, o.schema_name) == (CATALOG, SCHEMA) for o in result.objects)


def test_columns_keep_ordinal_order_full_type_and_null_for_empty_comment() -> None:
    result = _read(FakeWarehouse())

    orders = result.get("orders")
    assert isinstance(orders.columns, tuple)
    assert [(c.name, c.data_type, c.comment) for c in orders.columns] == [
        ("id", "BIGINT", "row id"),
        ("amount", "DECIMAL(10,2)", None),
        ("status", "STRING", None),
    ]
    assert [c.name for c in result.get("orders_v").columns] == ["id", "amount"]


def test_view_definition_is_set_for_views_only() -> None:
    result = _read(FakeWarehouse())

    view = result.get("orders_v")
    assert view.view_definition == VIEW_SQL
    assert view.is_view is True
    for name in ("orders", "Mixed_Case", "hidden_v"):
        obj = result.get(name)
        assert obj.view_definition is None, name
        assert obj.is_view is False, name


def test_lookup_on_the_read_result_ignores_case() -> None:
    result = _read(FakeWarehouse())

    assert result.get("mixed_case").name == "Mixed_Case"
    assert result.get("ORDERS_V").name == "orders_v"
    assert result.get("absent") is None


# --- acceptance 3: warehouse errors become Unreadable, never raise ---

PERMISSION_TEXT = "PERMISSION_DENIED: User does not have USE SCHEMA on Schema 'cat_a.sch_a'."


@pytest.mark.parametrize("fail_on", VIEWS)
@pytest.mark.parametrize(
    "error",
    [
        databricks_exc.ServerOperationError(f"{PERMISSION_TEXT}\nSQLSTATE: 42501"),
        databricks_exc.Error(f"{PERMISSION_TEXT}\nsecond line"),
    ],
    ids=["server_operation_error", "base_warehouse_error"],
)
def test_warehouse_error_returns_unreadable_with_class_and_first_line(
    fail_on: str, error: Exception, log_lines: list[str]
) -> None:
    from sema.models.schema_metadata import Unreadable

    result = _read(FakeWarehouse(fail_on=fail_on, error=error))

    assert isinstance(result, Unreadable)
    assert (result.catalog, result.schema_name) == (CATALOG, SCHEMA)
    assert type(error).__name__ in result.reason
    assert PERMISSION_TEXT in result.reason
    assert "\n" not in result.reason
    warnings = [line for line in log_lines if "WARNING" in line]
    assert len(warnings) == 1
    assert SCHEMA in warnings[0]


def test_malformed_rows_are_not_swallowed() -> None:
    warehouse = FakeWarehouse(raw_rows={"columns": [("only_one_field",)]})

    with pytest.raises((ValueError, IndexError, TypeError, KeyError)):
        _read(warehouse)


# --- acceptance 4: quotes and backslashes in names cannot break the SQL ---


@pytest.mark.parametrize(
    "schema",
    ["o'brien", "x' OR '1'='1", "trailing\\"],
    ids=["quote", "injection", "backslash"],
)
def test_schema_name_is_escaped_as_one_literal(schema: str) -> None:
    warehouse = FakeWarehouse()

    _read(warehouse, schema=schema)

    assert len(warehouse.statements) == 3
    for sql, params in warehouse.statements:
        tree = _parse(sql)
        assert tree.find(exp.Or) is None, sql
        equalities = _where_equalities(tree, params)
        assert equalities.get("table_schema") == [schema], sql
        assert equalities.get("table_catalog") == [CATALOG], sql


def test_catalog_name_with_quote_is_escaped() -> None:
    warehouse = FakeWarehouse()
    catalog = "it's"

    _read(warehouse, catalog=catalog)

    for sql, params in warehouse.statements:
        assert _where_equalities(_parse(sql), params).get("table_catalog") == [catalog], sql


# --- acceptance 5: dialect ---


def test_databricks_connector_reports_its_sql_dialect() -> None:
    assert DatabricksConnector.sql_dialect == "databricks"
