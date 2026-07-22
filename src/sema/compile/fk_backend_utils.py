"""S1-08 — Databricks/Delta SQL builders for the FK-closed OMOP shape (generic).

The DuckDB builders live in :mod:`sema.compile.fk_closed_compiler_utils`; these
are their Delta siblings. Databricks has NO multi-table transaction and no client
temp table in a SQL-warehouse session, so the child scoped-swap uses Delta's
atomic ``INSERT INTO ... REPLACE WHERE <scope>`` (mirrors the §1.5(b) staging
write) instead of ``BEGIN/DELETE/INSERT/COMMIT``. The parent is rebuilt with
``CREATE OR REPLACE TABLE ... AS SELECT`` (atomic on Delta).

R29-scanned generic spine: every physical column name arrives via a spec from the
policy boundary; nothing here names ``person``/``condition-occurrence``/OMOP.
"""

from __future__ import annotations

from typing import Sequence

import sqlglot.expressions as exp

from sema.compile.fk_closed_compiler_utils import (
    ChildSourceSpec,
    ChildTableSpec,
    ParentTableSpec,
    RegistryJoinSpec,
)

# DuckDB column types -> Databricks/Delta equivalents.
_TYPE_MAP = {"VARCHAR": "STRING", "BIGINT": "BIGINT", "DATE": "DATE"}


def backtick(ident: str) -> str:
    return f"`{ident}`"


def qualified(schema: str, table: str) -> str:
    return f"{backtick(schema)}.{backtick(table)}"


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def databricks_replace_parent_sql(
    parent: ParentTableSpec, registry: RegistryJoinSpec
) -> str:
    """Rebuild the FK-target from the WHOLE bridged registry, distinct ids."""
    return (
        f"CREATE OR REPLACE TABLE {qualified(parent.schema, parent.table)} "
        f"USING DELTA AS "
        f"SELECT DISTINCT {backtick(registry.id_column)} AS {backtick(parent.id_column)} "
        f"FROM {qualified(registry.schema, registry.table)}"
    )


def databricks_create_child_table_sql(child: ChildTableSpec) -> str:
    """``CREATE TABLE IF NOT EXISTS ... USING DELTA`` for the child columns."""
    types = child.column_types()
    cols = ",\n  ".join(
        f"{backtick(c)} {_TYPE_MAP[types[c]]}" for c in child.column_order()
    )
    return (
        f"CREATE TABLE IF NOT EXISTS {qualified(child.schema, child.table)} "
        f"(\n  {cols}\n) USING DELTA"
    )


def databricks_child_scope_predicate(child: ChildTableSpec, source_schema: str) -> str:
    """Inline-literal scope predicate (the value is an internal schema id)."""
    return f"{backtick(child.scope_schema_column)} = {_sql_string(source_schema)}"


def databricks_replace_where_child_sql(
    child: ChildTableSpec, source: ChildSourceSpec, select: exp.Select
) -> str:
    """Atomic Delta replace of one source scope with the compiled child SELECT."""
    predicate = databricks_child_scope_predicate(child, source.schema)
    body = select.sql(dialect="databricks")
    return (
        f"INSERT INTO {qualified(child.schema, child.table)} "
        f"REPLACE WHERE ({predicate})\n{body}"
    )


def databricks_child_scope_count_sql(
    child: ChildTableSpec, source_schema: str
) -> str:
    predicate = databricks_child_scope_predicate(child, source_schema)
    return f"SELECT COUNT(*) FROM {qualified(child.schema, child.table)} WHERE {predicate}"


def databricks_count_all_sql(parent: ParentTableSpec) -> str:
    return f"SELECT COUNT(*) FROM {qualified(parent.schema, parent.table)}"


def databricks_orphan_fk_count_sql(
    parent: ParentTableSpec, child: ChildTableSpec
) -> str:
    """Count child rows whose FK is absent from the parent (must be 0 at rest)."""
    return (
        f"SELECT COUNT(*) FROM {qualified(child.schema, child.table)} c "
        f"LEFT JOIN {qualified(parent.schema, parent.table)} p "
        f"ON c.{backtick(child.fk_column)} = p.{backtick(parent.id_column)} "
        f"WHERE p.{backtick(parent.id_column)} IS NULL"
    )


def databricks_missing_key_count_sql(source: ChildSourceSpec) -> str:
    """Count source rows with a blank patient key (the review disposition, D5)."""
    return (
        f"SELECT COUNT(*) FROM {qualified(source.schema, source.table)} "
        f"WHERE TRIM(COALESCE({backtick(source.patient_key_column)}, '')) = ''"
    )


def databricks_column_null_counts_sql(
    child: ChildTableSpec, fields: Sequence[str], source_schema: str
) -> str:
    """Per-column NULL counts for the required fields, scoped to one study."""
    sums = ", ".join(
        f"SUM(CASE WHEN {backtick(c)} IS NULL THEN 1 ELSE 0 END)" for c in fields
    )
    predicate = databricks_child_scope_predicate(child, source_schema)
    return (
        f"SELECT {sums} FROM {qualified(child.schema, child.table)} WHERE {predicate}"
    )
