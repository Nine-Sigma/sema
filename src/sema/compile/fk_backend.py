"""S1-08 — multi-warehouse backend for the FK-closed OMOP shape.

Mirrors :class:`sema.compile.staging_backend.StagingBackend`: the FK-closed
compiler (S1-06) is written once and runs against DuckDB (the fit engine) or
Databricks (the live engine). The only things that differ per warehouse are
identifier quoting and the child scoped-swap mechanic — DuckDB uses a temp-build
+ transactional scoped-swap; Databricks uses Delta's atomic ``INSERT ... REPLACE
WHERE`` (no client temp table, no multi-statement transaction on a SQL
warehouse). Injecting the backend keeps the compiler free of warehouse branches.

R29-scanned generic spine: this module names no showcase literal. Physical names
arrive via the spec dataclasses from the policy boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, Sequence

import sqlglot.expressions as exp

from sema.compile.fk_backend_utils import (
    databricks_child_scope_count_sql,
    databricks_column_null_counts_sql,
    databricks_count_all_sql,
    databricks_create_child_table_sql,
    databricks_missing_key_count_sql,
    databricks_orphan_fk_count_sql,
    databricks_replace_parent_sql,
    databricks_replace_where_child_sql,
)
from sema.compile.fk_backend_utils import qualified as databricks_qualified
from sema.compile.fk_closed_compiler_utils import (
    ChildSourceSpec,
    ChildTableSpec,
    ParentTableSpec,
    RegistryJoinSpec,
    create_child_table_sql,
    delete_child_scope_sql,
    insert_child_from_temp_sql,
    missing_key_count_sql,
    orphan_fk_count_sql,
    replace_parent_sql,
)

__all__ = [
    "DATABRICKS_FK_BACKEND",
    "DUCKDB_FK_BACKEND",
    "FkBackend",
    "FkCursor",
    "fk_backend_for",
]

_TEMP_TABLE = "_sema_fk_child_build"


class FkCursor(Protocol):
    """The DuckDB-connection / Databricks-cursor surface the backends drive."""

    def execute(self, sql: str, parameters: Sequence[Any] = ...) -> Any: ...

    def fetchone(self) -> Any: ...


class FkBackend(ABC):
    """Render + write the FK-closed parent/child for one warehouse dialect."""

    dialect: str

    @abstractmethod
    def qualified(self, schema: str, table: str) -> str:
        """Dialect-quoted ``schema.table`` reference."""

    @abstractmethod
    def replace_parent(
        self, cur: FkCursor, parent: ParentTableSpec, registry: RegistryJoinSpec
    ) -> None:
        """Rebuild the FK-target from the whole registry (distinct ids)."""

    @abstractmethod
    def write_child(
        self,
        cur: FkCursor,
        child: ChildTableSpec,
        source: ChildSourceSpec,
        select: exp.Select,
    ) -> int:
        """Create the child table, replace one source scope, return rows written."""

    @abstractmethod
    def _orphan_sql(self, parent: ParentTableSpec, child: ChildTableSpec) -> str: ...

    @abstractmethod
    def _missing_key_sql(self, source: ChildSourceSpec) -> str: ...

    @abstractmethod
    def _count_all_sql(self, parent: ParentTableSpec) -> str: ...

    @abstractmethod
    def _child_scope_count_sql(self, child: ChildTableSpec, source_schema: str) -> str: ...

    @abstractmethod
    def _null_counts_sql(
        self, child: ChildTableSpec, fields: Sequence[str], source_schema: str
    ) -> str: ...

    def orphan_fk_count(
        self, cur: FkCursor, parent: ParentTableSpec, child: ChildTableSpec
    ) -> int:
        return self._scalar(cur, self._orphan_sql(parent, child))

    def missing_key_count(self, cur: FkCursor, source: ChildSourceSpec) -> int:
        return self._scalar(cur, self._missing_key_sql(source))

    def count_all(self, cur: FkCursor, parent: ParentTableSpec) -> int:
        return self._scalar(cur, self._count_all_sql(parent))

    def child_scope_count(
        self, cur: FkCursor, child: ChildTableSpec, source_schema: str
    ) -> int:
        return self._scalar(cur, self._child_scope_count_sql(child, source_schema))

    def column_null_counts(
        self,
        cur: FkCursor,
        child: ChildTableSpec,
        fields: Sequence[str],
        source_schema: str,
    ) -> dict[str, int]:
        if not fields:
            return {}
        row = self._fetch(cur, self._null_counts_sql(child, fields, source_schema))
        return {c: int(row[i] or 0) for i, c in enumerate(fields)}

    def _scalar(self, cur: FkCursor, sql: str) -> int:
        row = self._fetch(cur, sql)
        return int(row[0]) if row else 0

    @staticmethod
    def _fetch(cur: FkCursor, sql: str) -> Any:
        cur.execute(sql)
        return cur.fetchone()


class DuckdbFkBackend(FkBackend):
    """Temp-build + transactional scoped-swap on DuckDB (the fit/default engine)."""

    dialect = "duckdb"

    def qualified(self, schema: str, table: str) -> str:
        return f'"{schema}"."{table}"'

    def replace_parent(
        self, cur: FkCursor, parent: ParentTableSpec, registry: RegistryJoinSpec
    ) -> None:
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{parent.schema}"')
        cur.execute(replace_parent_sql(parent, registry))

    def write_child(
        self,
        cur: FkCursor,
        child: ChildTableSpec,
        source: ChildSourceSpec,
        select: exp.Select,
    ) -> int:
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{child.schema}"')
        cur.execute(create_child_table_sql(child))
        cur.execute(
            f"CREATE OR REPLACE TEMP TABLE {_TEMP_TABLE} AS "
            f"{select.sql(dialect='duckdb')}"
        )
        cur.execute("BEGIN TRANSACTION")
        cur.execute(delete_child_scope_sql(child), [source.schema])
        cur.execute(insert_child_from_temp_sql(child, _TEMP_TABLE))
        cur.execute("COMMIT")
        cur.execute(f"DROP TABLE IF EXISTS {_TEMP_TABLE}")
        return self.child_scope_count(cur, child, source.schema)

    def _orphan_sql(self, parent: ParentTableSpec, child: ChildTableSpec) -> str:
        return orphan_fk_count_sql(parent, child)

    def _missing_key_sql(self, source: ChildSourceSpec) -> str:
        return missing_key_count_sql(source)

    def _count_all_sql(self, parent: ParentTableSpec) -> str:
        return f'SELECT COUNT(*) FROM "{parent.schema}"."{parent.table}"'

    def _child_scope_count_sql(self, child: ChildTableSpec, source_schema: str) -> str:
        return (
            f'SELECT COUNT(*) FROM "{child.schema}"."{child.table}" '
            f"WHERE \"{child.scope_schema_column}\" = '{source_schema}'"
        )

    def _null_counts_sql(
        self, child: ChildTableSpec, fields: Sequence[str], source_schema: str
    ) -> str:
        sums = ", ".join(
            f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END)' for c in fields
        )
        return (
            f'SELECT {sums} FROM "{child.schema}"."{child.table}" '
            f"WHERE \"{child.scope_schema_column}\" = '{source_schema}'"
        )


class DatabricksFkBackend(FkBackend):
    """Atomic Delta ``INSERT ... REPLACE WHERE`` scoped-swap on Databricks (prod)."""

    dialect = "databricks"

    def qualified(self, schema: str, table: str) -> str:
        return databricks_qualified(schema, table)

    def replace_parent(
        self, cur: FkCursor, parent: ParentTableSpec, registry: RegistryJoinSpec
    ) -> None:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS `{parent.schema}`")
        cur.execute(databricks_replace_parent_sql(parent, registry))

    def write_child(
        self,
        cur: FkCursor,
        child: ChildTableSpec,
        source: ChildSourceSpec,
        select: exp.Select,
    ) -> int:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS `{child.schema}`")
        cur.execute(databricks_create_child_table_sql(child))
        cur.execute(databricks_replace_where_child_sql(child, source, select))
        return self.child_scope_count(cur, child, source.schema)

    def _orphan_sql(self, parent: ParentTableSpec, child: ChildTableSpec) -> str:
        return databricks_orphan_fk_count_sql(parent, child)

    def _missing_key_sql(self, source: ChildSourceSpec) -> str:
        return databricks_missing_key_count_sql(source)

    def _count_all_sql(self, parent: ParentTableSpec) -> str:
        return databricks_count_all_sql(parent)

    def _child_scope_count_sql(self, child: ChildTableSpec, source_schema: str) -> str:
        return databricks_child_scope_count_sql(child, source_schema)

    def _null_counts_sql(
        self, child: ChildTableSpec, fields: Sequence[str], source_schema: str
    ) -> str:
        return databricks_column_null_counts_sql(child, fields, source_schema)


DUCKDB_FK_BACKEND: FkBackend = DuckdbFkBackend()
DATABRICKS_FK_BACKEND: FkBackend = DatabricksFkBackend()

_BACKENDS: dict[str, FkBackend] = {
    "duckdb": DUCKDB_FK_BACKEND,
    "databricks": DATABRICKS_FK_BACKEND,
}


def fk_backend_for(name: str) -> FkBackend:
    """Resolve an FK backend by name (``duckdb`` | ``databricks``)."""
    try:
        return _BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown fk backend: {name!r}") from None
