"""A neutral warehouse for running `run_build` without a warehouse or a model.

DuckDB stands in for the warehouse and answers the connector's own SQL; a
deterministic client stands in for the model. Two schemas share one catalog: the
main schema holds every join shape, and the decoy schema holds a table with the
same name and columns as a main-schema table that joins.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import duckdb
import sqlglot
from pydantic import SecretStr

from sema.llm_client import LLMClient, SynonymExpansion, TableSummary, VocabularyDetection
from sema.models.config import BuildConfig, DatabricksConfig, Neo4jConfig
from sema.models.stages import (
    StageAResult,
    StageBBatchResult,
    StageBColumnResult,
    StageCBatchResult,
    StageCResult,
)

CATALOG = "fixcat"
MAIN_SCHEMA = "shop"
DECOY_SCHEMA = "shop_mirror"
HOST = "fixture-host.example"
COMPAT_SCHEMA = "warehouse_information_schema"


@dataclass(frozen=True)
class DeclaredKey:
    columns: tuple[str, ...]
    referenced_table: str
    referenced_columns: tuple[str, ...]


@dataclass(frozen=True)
class FixtureTable:
    name: str
    columns: tuple[tuple[str, str], ...]
    primary_key: tuple[str, ...]
    rows: str
    declared_keys: tuple[DeclaredKey, ...] = ()

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.columns)


@dataclass(frozen=True)
class FixtureJoin:
    """A single-column join; `None` on the referenced side means any relation."""

    referencing_table: str
    referencing_column: str
    referenced_table: str | None
    referenced_column: str | None


_ACCOUNT = FixtureTable(
    "account",
    (("account_id", "INTEGER"), ("customer_id", "INTEGER"), ("opened_on", "DATE")),
    ("account_id",),
    "SELECT i, 1 + (i % 200), DATE '2020-01-01' + CAST(i AS INT) FROM range(1, 401) t(i)",
    (DeclaredKey(("customer_id",), "customer", ("customer_id",)),),
)

MAIN_TABLES: tuple[FixtureTable, ...] = (
    FixtureTable(
        "customer",
        (("customer_id", "INTEGER"), ("customer_name", "VARCHAR"), ("region_code", "VARCHAR")),
        ("customer_id",),
        "SELECT i, 'cust ' || i, CASE i % 3 WHEN 0 THEN 'N' WHEN 1 THEN 'S' ELSE 'W' END "
        "FROM range(1, 201) t(i)",
    ),
    _ACCOUNT,
    FixtureTable(
        "transfer",
        (("transfer_id", "INTEGER"), ("from_account_id", "INTEGER"),
         ("to_account_id", "INTEGER"), ("amount", "DOUBLE"), ("transfer_date", "DATE")),
        ("transfer_id",),
        "SELECT i, 1 + (i % 400), 1 + ((i * 7) % 400), i * 1.5, "
        "DATE '2024-01-01' + CAST(i % 300 AS INT) FROM range(1, 2001) t(i)",
        (DeclaredKey(("from_account_id",), "account", ("account_id",)),
         DeclaredKey(("to_account_id",), "account", ("account_id",))),
    ),
    FixtureTable(
        "product",
        (("product_id", "INTEGER"), ("product_name", "VARCHAR"), ("category_code", "VARCHAR")),
        ("product_id",),
        "SELECT i, 'prod ' || i, CASE i % 4 WHEN 0 THEN 'A' WHEN 1 THEN 'B' WHEN 2 THEN 'C' "
        "ELSE 'D' END FROM range(1, 51) t(i)",
    ),
    FixtureTable(
        "supplier",
        (("supplier_id", "INTEGER"), ("supplier_name", "VARCHAR")),
        ("supplier_id",),
        "SELECT i, 'sup ' || i FROM range(1, 21) t(i)",
    ),
    FixtureTable(
        "memo",
        (("memo_id", "INTEGER"), ("vendor_ref", "INTEGER"), ("body", "VARCHAR")),
        ("memo_id",),
        "SELECT i, 1 + (i % 20), 'memo ' || i FROM range(1, 61) t(i)",
        (DeclaredKey(("vendor_ref",), "supplier", ("supplier_id",)),),
    ),
    FixtureTable(
        "product_supplier",
        (("product_id", "INTEGER"), ("supplier_id", "INTEGER")),
        ("product_id", "supplier_id"),
        "SELECT DISTINCT 1 + (i % 50), 1 + ((i * 3) % 20) FROM range(1, 301) t(i)",
    ),
    FixtureTable(
        "order_header",
        (("order_id", "INTEGER"), ("customer_id", "INTEGER"), ("order_date", "DATE"),
         ("legacy_customer_id", "INTEGER")),
        ("order_id",),
        "SELECT i, 1 + (i % 200), DATE '2024-01-01' + CAST(i % 300 AS INT), 5000 + i "
        "FROM range(1, 501) t(i)",
    ),
    FixtureTable(
        "order_line",
        (("order_id", "INTEGER"), ("line_no", "INTEGER"), ("product_id", "INTEGER"),
         ("qty", "INTEGER")),
        ("order_id", "line_no"),
        "SELECT o, l, 1 + ((o * l) % 50), l FROM range(1, 501) a(o), range(1, 4) b(l)",
    ),
    FixtureTable(
        "shipment_line",
        (("shipment_id", "INTEGER"), ("order_id", "INTEGER"), ("line_no", "INTEGER"),
         ("shipped_qty", "INTEGER")),
        ("shipment_id",),
        "SELECT o * 10 + l, o, l, l FROM range(1, 401) a(o), range(1, 3) b(l)",
        (DeclaredKey(("order_id", "line_no"), "order_line", ("order_id", "line_no")),
         DeclaredKey(("order_id",), "order_header", ("order_id",))),
    ),
    FixtureTable(
        "price_period",
        (("product_id", "INTEGER"), ("valid_from", "DATE"), ("valid_to", "DATE"),
         ("unit_price", "DOUBLE")),
        ("product_id", "valid_from"),
        "SELECT p, DATE '2023-01-01' + CAST(k * 180 AS INT), "
        "DATE '2023-01-01' + CAST(k * 180 + 179 AS INT), p * k * 1.0 "
        "FROM range(1, 51) a(p), range(0, 5) b(k)",
    ),
)

DECOY_TABLES: tuple[FixtureTable, ...] = (
    FixtureTable(_ACCOUNT.name, _ACCOUNT.columns, _ACCOUNT.primary_key, _ACCOUNT.rows),
)

SCHEMA_TABLES: dict[str, tuple[FixtureTable, ...]] = {
    MAIN_SCHEMA: MAIN_TABLES,
    DECOY_SCHEMA: DECOY_TABLES,
}

COLUMNS_BY_TABLE: dict[str, tuple[str, ...]] = {
    table.name: table.column_names for tables in SCHEMA_TABLES.values() for table in tables
}

EXPECTED_DETECTED_JOINS: tuple[FixtureJoin, ...] = (
    FixtureJoin("account", "customer_id", "customer", "customer_id"),
    FixtureJoin("order_header", "customer_id", "customer", "customer_id"),
    FixtureJoin("order_line", "order_id", "order_header", "order_id"),
    FixtureJoin("shipment_line", "order_id", "order_header", "order_id"),
    FixtureJoin("transfer", "from_account_id", "account", "account_id"),
    FixtureJoin("transfer", "to_account_id", "account", "account_id"),
    FixtureJoin("product_supplier", "product_id", "product", "product_id"),
    FixtureJoin("product_supplier", "supplier_id", "supplier", "supplier_id"),
    FixtureJoin("order_line", "product_id", "product", "product_id"),
    FixtureJoin("price_period", "product_id", "product", "product_id"),
)

# Declared on a table that sorts before the one it references, with no name affinity.
UNDETECTABLE_DECLARED_JOIN = FixtureJoin("memo", "vendor_ref", "supplier", "supplier_id")

DECOY_JOINS: tuple[FixtureJoin, ...] = (
    FixtureJoin("order_line", "qty", None, None),
    FixtureJoin("order_line", "line_no", None, None),
    FixtureJoin("order_header", "legacy_customer_id", None, None),
    FixtureJoin("shipment_line", "order_id", "order_line", "order_id"),
    FixtureJoin("order_line", "product_id", "price_period", "product_id"),
)

_SYSTEM_INFORMATION_SCHEMA = "`system`.`information_schema`."
_SHOW_TABLES = re.compile(r"^SHOW TABLES IN `(?P<catalog>[^`]+)`\.`(?P<schema>[^`]+)`$")
_COMPAT_VIEWS = {
    "columns": "SELECT *, COLUMN_COMMENT AS comment, data_type AS full_data_type "
               "FROM information_schema.columns",
    "tables": "SELECT * FROM information_schema.tables",
    "table_constraints": "SELECT * FROM information_schema.table_constraints",
    "referential_constraints": "SELECT * FROM information_schema.referential_constraints",
    "key_column_usage": "SELECT * FROM information_schema.key_column_usage",
}


def _create_table_sql(schema: str, table: FixtureTable) -> str:
    parts = [f"{name} {data_type}" for name, data_type in table.columns]
    parts.append(f"PRIMARY KEY ({', '.join(table.primary_key)})")
    parts.extend(
        f"FOREIGN KEY ({', '.join(key.columns)}) REFERENCES "
        f"{schema}.{key.referenced_table} ({', '.join(key.referenced_columns)})"
        for key in table.declared_keys
    )
    return f"CREATE TABLE {schema}.{table.name} ({', '.join(parts)})"


def build_warehouse() -> duckdb.DuckDBPyConnection:
    """The fixture warehouse with every table, key and row; DuckDB enforces the keys."""
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH ':memory:' AS {CATALOG}")
    con.execute(f"USE {CATALOG}")
    for schema, tables in SCHEMA_TABLES.items():
        con.execute(f"CREATE SCHEMA {schema}")
        for table in tables:
            con.execute(_create_table_sql(schema, table))
            con.execute(f"INSERT INTO {schema}.{table.name} {table.rows}")
    con.execute("USE memory")
    con.execute(f"CREATE SCHEMA {COMPAT_SCHEMA}")
    for view, body in _COMPAT_VIEWS.items():
        con.execute(f"CREATE VIEW {COMPAT_SCHEMA}.{view} AS {body}")
    return con


class FakeCursor:
    def __init__(self, con: duckdb.DuckDBPyConnection, unanswered: list[str]) -> None:
        self._con = con
        self._unanswered = unanswered
        self._rows: list[tuple[Any, ...]] = []
        self.description: list[tuple[Any, ...]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, query: str) -> None:
        self.description = []
        show_tables = _SHOW_TABLES.match(query.strip())
        if show_tables:
            # DuckDB's SHOW TABLES returns names only, not (database, tableName, isTemporary).
            self._rows = [
                (schema, table, False) for schema, table in self._con.execute(
                    "SELECT table_schema, table_name FROM information_schema.tables "
                    "WHERE table_catalog = ? AND table_schema = ? ORDER BY table_name",
                    [show_tables["catalog"], show_tables["schema"]],
                ).fetchall()
            ]
            return
        if "`column_tags`" in query:
            # DuckDB has no column tags, so the honest answer is none.
            self._rows = []
            return
        self._run_translated(query)

    def _run_translated(self, query: str) -> None:
        local = query.replace(_SYSTEM_INFORMATION_SCHEMA, f"`memory`.`{COMPAT_SCHEMA}`.")
        try:
            duck = sqlglot.transpile(local, read="databricks", write="duckdb")[0]
            result = self._con.execute(duck)
            self._rows = result.fetchall()
            self.description = list(result.description or [])
        except Exception as exc:
            self._unanswered.append(f"{type(exc).__name__}: {query}")
            raise

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class FakeConnection:
    def __init__(self, warehouse: duckdb.DuckDBPyConnection) -> None:
        self._warehouse = warehouse
        self.unanswered: list[str] = []

    def cursor(self) -> FakeCursor:
        return FakeCursor(self._warehouse.cursor(), self.unanswered)


class FakeLLMClient(LLMClient):
    """Names each entity after its table and each property after its column."""

    def invoke(  # type: ignore[override]
        self, prompt: str, schema: Any, *, table_ref: str = "", stage_name: str = "",
        simplified_prompt: str | None = None,
    ) -> Any:
        table = table_ref.rstrip("/").rsplit("/", 1)[-1]
        entity = table.replace("_", " ").title()
        if schema is StageAResult:
            return StageAResult(
                primary_entity=entity, grain_hypothesis=f"one row per {table}", confidence=0.9,
            )
        if schema is StageBBatchResult:
            return StageBBatchResult(columns=[
                StageBColumnResult(
                    column=column, canonical_property_label=column.replace("_", " "),
                    semantic_type="identifier" if column.endswith("_id") else "attribute",
                    confidence=0.9,
                )
                for column in COLUMNS_BY_TABLE.get(table, ()) if column in prompt
            ])
        if schema is StageCBatchResult:
            return StageCBatchResult(columns=[])
        if schema is StageCResult:
            return StageCResult(column="")
        if schema is TableSummary:
            return TableSummary(entity_name=entity)
        if schema is VocabularyDetection:
            return VocabularyDetection(vocabulary="unknown", confidence=0.0)
        if schema is SynonymExpansion:
            return SynonymExpansion(synonyms=[])
        raise AssertionError(f"unexpected schema {schema}")


def make_fake_llm_client() -> FakeLLMClient:
    return FakeLLMClient(llm=MagicMock())


def fixture_build_config(neo4j: Neo4jConfig) -> BuildConfig:
    return BuildConfig(
        catalog=CATALOG,
        schemas=list(SCHEMA_TABLES),
        skip_embeddings=True,
        table_workers=1,
        databricks=DatabricksConfig(
            host=f"https://{HOST}", http_path="/fixture", token=SecretStr("fixture"),
        ),
        neo4j=neo4j,
    )


@contextmanager
def fixture_environment() -> Iterator[list[FakeConnection]]:
    """Route every connector to the fixture warehouse and every model call to the fake.

    Yields the connections opened, so a caller can read the statements left unanswered.
    """
    warehouse = build_warehouse()
    connections: list[FakeConnection] = []

    def connect(**_: Any) -> FakeConnection:
        connection = FakeConnection(warehouse)
        connections.append(connection)
        return connection

    with (
        patch("sema.connectors.databricks.sql_connect", connect),
        patch("sema.pipeline.build.LLMClientFactory.create", lambda _self: make_fake_llm_client()),
    ):
        yield connections
    warehouse.close()
