"""Multi-warehouse staging backend (US-013).

The Slice-0 chain is written once and runs against more than one warehouse: the
same compiled SELECT (built by :mod:`sema.compile.compiler_utils`) is rendered
per dialect and written via a :class:`StagingBackend` strategy. DuckDB (dev,
``~/.sema/poc.duckdb``) uses a temp-build + scoped-swap; Databricks (prod,
``workspace.*``) uses Delta's atomic ``INSERT ... REPLACE WHERE``. Injecting the
backend keeps the chain (``run_fit``) free of warehouse branches — the only new
input is *which* strategy to use.

R29-scanned generic spine: this module names no showcase literal. Physical
column names arrive via :class:`StagingColumns` from the policy boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, Sequence

import sqlglot.expressions as exp

from sema.compile.compiler_utils import (
    SourceTableSpec,
    StagingColumns,
    count_scope_sql,
    create_staging_table_sql,
    delete_scope_sql,
    insert_from_temp_sql,
)
from sema.compile.staging_backend_utils import (
    databricks_count_scope_sql,
    databricks_create_staging_table_sql,
    databricks_replace_where_sql,
    databricks_scope_predicate,
    qualified as databricks_qualified,
)

__all__ = [
    "DATABRICKS_BACKEND",
    "DUCKDB_BACKEND",
    "StagingBackend",
    "StagingCursor",
    "staging_backend_for",
]

_TEMP_TABLE = "_sema_staging_build"


class StagingCursor(Protocol):
    """The DuckDB-connection / Databricks-cursor surface the backends drive."""

    def execute(self, sql: str, parameters: Sequence[Any] = ...) -> Any: ...

    def fetchone(self) -> Any: ...

    def fetchall(self) -> list[Any]: ...


class StagingBackend(ABC):
    """Render + write the §1.5(b) staging table for one warehouse dialect."""

    dialect: str

    @abstractmethod
    def quote(self, ident: str) -> str:
        """Dialect-quoted identifier."""

    @abstractmethod
    def qualified(self, schema: str, table: str) -> str:
        """Dialect-quoted ``schema.table`` reference."""

    @abstractmethod
    def scope_predicate(
        self, columns: StagingColumns, source_schema: str, source_table: str
    ) -> tuple[str, list[Any]]:
        """``(predicate_sql, params)`` selecting one source scope."""

    @abstractmethod
    def write_staging(
        self,
        cursor: StagingCursor,
        select: exp.Select,
        *,
        columns: StagingColumns,
        source: SourceTableSpec,
        staging_schema: str,
        staging_table: str,
    ) -> int:
        """Create the staging table, write the scope, return rows written."""

    def query(
        self, cursor: StagingCursor, sql: str, params: Sequence[Any] | None = None
    ) -> list[Any]:
        self._exec(cursor, sql, params)
        return list(cursor.fetchall())

    def fetch_one(
        self, cursor: StagingCursor, sql: str, params: Sequence[Any] | None = None
    ) -> Any:
        self._exec(cursor, sql, params)
        return cursor.fetchone()

    def run(
        self, cursor: StagingCursor, sql: str, params: Sequence[Any] | None = None
    ) -> None:
        self._exec(cursor, sql, params)

    @staticmethod
    def _exec(
        cursor: StagingCursor, sql: str, params: Sequence[Any] | None
    ) -> None:
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)


class DuckdbStagingBackend(StagingBackend):
    """Temp-build + scoped-swap write on DuckDB (the dev/default backend)."""

    dialect = "duckdb"

    def quote(self, ident: str) -> str:
        return f'"{ident}"'

    def qualified(self, schema: str, table: str) -> str:
        return f'"{schema}"."{table}"'

    def scope_predicate(
        self, columns: StagingColumns, source_schema: str, source_table: str
    ) -> tuple[str, list[Any]]:
        predicate = (
            f'"{columns.source_schema}" = ? AND "{columns.source_table}" = ?'
        )
        return predicate, [source_schema, source_table]

    def write_staging(
        self,
        cursor: StagingCursor,
        select: exp.Select,
        *,
        columns: StagingColumns,
        source: SourceTableSpec,
        staging_schema: str,
        staging_table: str,
    ) -> int:
        self.run(cursor, f'CREATE SCHEMA IF NOT EXISTS "{staging_schema}"')
        self.run(cursor, create_staging_table_sql(columns, staging_schema, staging_table))
        self.run(
            cursor,
            f"CREATE OR REPLACE TEMP TABLE {_TEMP_TABLE} AS {select.sql(dialect='duckdb')}",
        )
        scope = [source.schema, source.table]
        self.run(cursor, "BEGIN TRANSACTION")
        self.run(cursor, delete_scope_sql(columns, staging_schema, staging_table), scope)
        self.run(
            cursor,
            insert_from_temp_sql(columns, staging_schema, staging_table, _TEMP_TABLE),
        )
        self.run(cursor, "COMMIT")
        self.run(cursor, f"DROP TABLE IF EXISTS {_TEMP_TABLE}")
        row = self.fetch_one(
            cursor, count_scope_sql(columns, staging_schema, staging_table), scope
        )
        return int(row[0]) if row else 0


class DatabricksStagingBackend(StagingBackend):
    """Atomic Delta ``INSERT ... REPLACE WHERE`` write on Databricks (prod)."""

    dialect = "databricks"

    def quote(self, ident: str) -> str:
        return f"`{ident}`"

    def qualified(self, schema: str, table: str) -> str:
        return databricks_qualified(schema, table)

    def scope_predicate(
        self, columns: StagingColumns, source_schema: str, source_table: str
    ) -> tuple[str, list[Any]]:
        return databricks_scope_predicate(columns, source_schema, source_table), []

    def write_staging(
        self,
        cursor: StagingCursor,
        select: exp.Select,
        *,
        columns: StagingColumns,
        source: SourceTableSpec,
        staging_schema: str,
        staging_table: str,
    ) -> int:
        self.run(cursor, f"CREATE SCHEMA IF NOT EXISTS `{staging_schema}`")
        self.run(
            cursor,
            databricks_create_staging_table_sql(columns, staging_schema, staging_table),
        )
        self.run(
            cursor,
            databricks_replace_where_sql(
                columns, staging_schema, staging_table, source, select
            ),
        )
        row = self.fetch_one(
            cursor,
            databricks_count_scope_sql(columns, staging_schema, staging_table, source),
        )
        return int(row[0]) if row else 0


DUCKDB_BACKEND: StagingBackend = DuckdbStagingBackend()
DATABRICKS_BACKEND: StagingBackend = DatabricksStagingBackend()

_BACKENDS: dict[str, StagingBackend] = {
    "duckdb": DUCKDB_BACKEND,
    "databricks": DATABRICKS_BACKEND,
}


def staging_backend_for(name: str) -> StagingBackend:
    """Resolve a backend by name (``duckdb`` | ``databricks``)."""
    try:
        return _BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown staging backend: {name!r}") from None
