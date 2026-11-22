"""DuckDB source connector: DTO-based extraction over DuckDB's catalog functions.

Every connection is read-only. Statements the build sends through
``execute_query`` are Databricks dialect and are translated here; the
connector's own ``duckdb_*()`` reads run as written.
"""

from __future__ import annotations

import fnmatch
import uuid
from typing import Any

import duckdb
import sqlglot

from sema.connectors.base import Connector
from sema.connectors.databricks import TableWorkItem
from sema.connectors.databricks_utils import declared_keys_from_rows, declared_keys_query
from sema.connectors.duckdb_utils import (
    DUCKDB_PLATFORM,
    assemble_table_assertions,
    bounded_read,
    duckdb_workspace,
    open_read_only,
    qualified_table,
    quote_ident,
)
from sema.log import logger
from sema.models.assertions import Assertion
from sema.models.config import DuckDBConfig, ProfilingConfig
from sema.models.extraction import (
    ExtractedColumn,
    ExtractedForeignKey,
    ExtractedSampleRows,
    ExtractedTopValues,
)
from sema.models.physical_key import CanonicalRef, PhysicalKey
from sema.pipeline.normalizer import AssertionNormalizer


class DuckDBConnector(Connector):
    sql_dialect = "duckdb"

    def __init__(self, config: DuckDBConfig, profiling: ProfilingConfig | None = None) -> None:
        self._config = config
        self.profiling = profiling or ProfilingConfig()
        self._run_id = str(uuid.uuid4())
        self._workspace = duckdb_workspace(config.path)
        # Opened eagerly so a mistyped path fails at construction, as Databricks does.
        self._connection: duckdb.DuckDBPyConnection | None = open_read_only(
            config.path, config.external_access,
        )

    # --- Connection ---

    def _execute(self, query: str, params: list[Any] | None = None) -> list[tuple[Any, ...]]:
        return self._cursor().execute(query, params or []).fetchall()

    def _cursor(self) -> duckdb.DuckDBPyConnection:
        if self._connection is None:
            raise RuntimeError("DuckDBConnector is closed")
        return self._connection

    def execute_query(self, query: str) -> list[tuple[Any, ...]]:
        """Run a Databricks-dialect read, translated to DuckDB."""
        return self._execute(sqlglot.transpile(query, read="databricks", write="duckdb")[0])

    def execute_query_with_columns(
        self, query: str, *, max_rows: int | None = None, timeout_s: float | None = None,
    ) -> tuple[list[str], list[tuple[Any, ...]]]:
        """Run a DuckDB-dialect read as written, untranslated, within the given bounds."""
        return bounded_read(self._cursor(), query, max_rows, timeout_s)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def get_datasource_ref(self) -> tuple[str, str, str]:
        ref = CanonicalRef.datasource(DUCKDB_PLATFORM, self._workspace)
        return ref, DUCKDB_PLATFORM, self._workspace

    # --- Discovery ---

    def list_catalogs(self) -> list[str]:
        rows = self._execute(
            "SELECT database_name FROM duckdb_databases() WHERE NOT internal ORDER BY 1"
        )
        return [row[0] for row in rows]

    def _require_catalog(self, catalog: str) -> None:
        attached = self.list_catalogs()
        if catalog not in attached:
            raise ValueError(f"Catalog {catalog!r} is not attached; available: {attached}")

    def _discover_schemas(self, catalog: str) -> list[str]:
        # Read from the relations: duckdb_schemas() marks the default schema `main` internal.
        rows = self._execute(
            "SELECT schema_name FROM duckdb_tables() WHERE database_name = ? AND NOT internal "
            "UNION SELECT schema_name FROM duckdb_views() WHERE database_name = ? AND NOT internal "
            "ORDER BY 1",
            [catalog, catalog],
        )
        return [row[0] for row in rows]

    def _discover_table_names(self, catalog: str, schema: str) -> list[str]:
        rows = self._execute(
            "SELECT table_name FROM duckdb_tables() "
            "WHERE database_name = ? AND schema_name = ? AND NOT internal "
            "ORDER BY table_name",
            [catalog, schema],
        )
        return [row[0] for row in rows]

    def discover_tables(
        self,
        catalog: str,
        schemas: list[str] | None = None,
        table_pattern: str | None = None,
    ) -> list[TableWorkItem]:
        self._require_catalog(catalog)
        return [
            TableWorkItem(
                catalog=catalog, schema=schema, table_name=table,
                fqn=CanonicalRef.build(
                    DUCKDB_PLATFORM, PhysicalKey(self._workspace, catalog, schema, table),
                ),
            )
            for schema in (schemas if schemas is not None else self._discover_schemas(catalog))
            for table in self._discover_table_names(catalog, schema)
            if not table_pattern or fnmatch.fnmatch(table, table_pattern)
        ]

    # --- Extraction ---

    def extract_table(self, work_item: TableWorkItem) -> list[Assertion]:
        normalizer = AssertionNormalizer(self._workspace, self._run_id, DUCKDB_PLATFORM)
        return assemble_table_assertions(self, work_item, normalizer, self.profiling)

    def extract(
        self,
        catalog: str,
        schemas: list[str] | None = None,
        table_pattern: str | None = None,
        **kwargs: Any,
    ) -> list[Assertion]:
        assertions: list[Assertion] = []
        for work_item in self.discover_tables(catalog, schemas, table_pattern):
            try:
                assertions.extend(self.extract_table(work_item))
            except duckdb.Error as exc:
                logger.warning("Failed to extract table {}: {}", work_item.fqn, exc)
        return assertions

    # --- DTO extraction ---

    def extract_columns_dto(
        self, catalog: str, schema: str, table: str,
    ) -> list[ExtractedColumn]:
        rows = self._execute(
            "SELECT column_name, data_type, is_nullable, comment FROM duckdb_columns() "
            "WHERE database_name = ? AND schema_name = ? AND table_name = ? "
            "ORDER BY column_index",
            [catalog, schema, table],
        )
        return [
            ExtractedColumn(
                name=name, table_name=table, catalog=catalog, schema=schema,
                data_type=data_type, nullable=bool(nullable), comment=comment,
            )
            for name, data_type, nullable, comment in rows
        ]

    def extract_foreign_keys_dto(
        self, catalog: str, schema: str, table: str,
    ) -> list[ExtractedForeignKey]:
        try:
            rows = self.execute_query(declared_keys_query(catalog, schema, table))
        except duckdb.Error as exc:
            logger.warning("Declared keys unreadable for {}.{}.{}: {}", catalog, schema, table, exc)
            return []
        return declared_keys_from_rows(catalog, schema, table, rows)

    def extract_sample_rows_dto(
        self, catalog: str, schema: str, table: str, limit: int = 10,
    ) -> ExtractedSampleRows:
        cursor = self._cursor().execute(
            f"SELECT * FROM {qualified_table(catalog, schema, table)} LIMIT ?", [limit],
        )
        column_names = [d[0] for d in cursor.description or []]
        rows = [dict(zip(column_names, map(str, row))) for row in cursor.fetchall()]
        return ExtractedSampleRows(
            table_name=table, catalog=catalog, schema=schema,
            rows=rows, column_names=column_names,
        )

    def extract_top_values_dto(
        self, catalog: str, schema: str, table: str, column: str, k: int = 20,
    ) -> ExtractedTopValues | None:
        """Top-K values, or ``None`` when the column is not categorical."""
        relation = qualified_table(catalog, schema, table)
        col = quote_ident(column)
        distinct_rows = self._execute(f"SELECT approx_count_distinct({col}) FROM {relation}")
        approx_distinct = int(distinct_rows[0][0]) if distinct_rows else 0
        if approx_distinct > self.profiling.categorical_threshold:
            return None
        rows = self._execute(
            f"SELECT {col}, COUNT(*) AS freq FROM {relation} "
            f"GROUP BY {col} ORDER BY freq DESC LIMIT ?",
            [k],
        )
        return ExtractedTopValues(
            column_name=column, table_name=table, catalog=catalog, schema=schema,
            values=[{"value": str(value), "frequency": freq} for value, freq in rows],
            approx_distinct=approx_distinct,
        )


class DuckDBConnectorFactory:
    """Creates one DuckDBConnector per worker."""

    def __init__(self, config: DuckDBConfig, profiling: ProfilingConfig | None = None) -> None:
        self._config = config
        self._profiling = profiling or ProfilingConfig()

    def create(self) -> DuckDBConnector:
        return DuckDBConnector(config=self._config, profiling=self._profiling)
