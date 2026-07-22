"""Databricks staging SQL builders (US-013).

R29-scanned generic spine: physical column names arrive via
:class:`StagingColumns`; this module names no showcase literal. The Databricks
staging write uses Delta's atomic ``INSERT INTO ... REPLACE WHERE <predicate>``
to replace exactly the (source_schema, source_table) partition — idempotent and
scoped without a temp table or a transaction coordinator (Databricks SQL
warehouses auto-commit each statement; ``REPLACE WHERE`` is atomic on Delta).
"""

from __future__ import annotations

import sqlglot.expressions as exp

from sema.compile.compiler_utils import (
    SourceTableSpec,
    StagingColumns,
    staging_column_order,
)

_BIGINT_FIELD = "target_concept_column"


def backtick(ident: str) -> str:
    return f"`{ident}`"


def qualified(schema: str, table: str) -> str:
    return f"{backtick(schema)}.{backtick(table)}"


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def databricks_create_staging_table_sql(
    columns: StagingColumns, schema: str, table: str
) -> str:
    """``CREATE TABLE IF NOT EXISTS ... USING DELTA`` for the §1.5(b) columns."""
    types = {
        getattr(columns, f): ("BIGINT" if f == _BIGINT_FIELD else "STRING")
        for f in _FIELDS
    }
    cols = ",\n  ".join(
        f"{backtick(name)} {types[name]}" for name in staging_column_order(columns)
    )
    return (
        f"CREATE TABLE IF NOT EXISTS {qualified(schema, table)} "
        f"(\n  {cols}\n) USING DELTA"
    )


def databricks_scope_predicate(
    columns: StagingColumns, source_schema: str, source_table: str
) -> str:
    """Inline-literal scope predicate (the values are internal identifiers)."""
    return (
        f"{backtick(columns.source_schema)} = {_sql_string(source_schema)} "
        f"AND {backtick(columns.source_table)} = {_sql_string(source_table)}"
    )


def databricks_replace_where_sql(
    columns: StagingColumns,
    schema: str,
    table: str,
    source: SourceTableSpec,
    select: exp.Select,
) -> str:
    """Atomic Delta replace of the source scope with the compiled SELECT."""
    predicate = databricks_scope_predicate(columns, source.schema, source.table)
    body = select.sql(dialect="databricks")
    return (
        f"INSERT INTO {qualified(schema, table)} REPLACE WHERE ({predicate})\n{body}"
    )


def databricks_count_scope_sql(
    columns: StagingColumns, schema: str, table: str, source: SourceTableSpec
) -> str:
    predicate = databricks_scope_predicate(columns, source.schema, source.table)
    return f"SELECT COUNT(*) FROM {qualified(schema, table)} WHERE {predicate}"


# Field names of StagingColumns in §1.5(b) order (mirrors compiler_utils).
_FIELDS: tuple[str, ...] = (
    "source_schema",
    "source_table",
    "source_row_ref",
    "source_patient_key",
    "source_value_column",
    "target_concept_column",
    "resolver_policy_ref",
    "vocab_release",
    "resolution_status",
    "no_map_reason",
    "status_column",
    "run_id",
)
