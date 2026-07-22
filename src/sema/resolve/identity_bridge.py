"""S1-08 — bridge the DuckDB-canonical identity registry into Databricks.

The identity registry (S1-01) is DuckDB-canonical: it is the single-writer store
where canonical ``entity_id``s are assigned. The live FK-closed materialization
(S1-08) runs in Databricks, where both the parent rebuild (``SELECT DISTINCT
entity_id``) and the child FK join read the registry via SQL — which only works
when the registry table is co-located with the target. So before a live run, the
registry's ``(source_namespace, source_entity_key, entity_id, ...)`` rows are
pushed to a Databricks Delta table, exactly as the value-mapping decisions are
inlined into the compiled SELECT.

The push is a full, idempotent replace (the DuckDB store stays canonical): a
re-run mirrors the current registry. Rows are inlined as batched ``VALUES`` (the
registry is small — one row per source patient), not streamed through a UC
Volume like the multi-million-row source pushes.

Generic (D6/R29): the frozen registry columns come from the registry contract;
nothing here names ``person``/OMOP.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from sema.resolve.identity_registry import DEFAULT_SCHEMA, DEFAULT_TABLE
from sema.resolve.identity_registry_utils import (
    FROZEN_COLUMNS,
    IdentityAssignment,
)

__all__ = [
    "DEFAULT_BATCH_ROWS",
    "bridge_identity_registry",
    "databricks_create_registry_table_sql",
    "databricks_insert_values_sql",
]

DEFAULT_BATCH_ROWS = 500

# DuckDB registry types -> Databricks/Delta. entity_id is the only non-string.
_DATABRICKS_TYPES = {c: ("BIGINT" if c == "entity_id" else "STRING") for c in FROZEN_COLUMNS}


class _Cursor(Protocol):
    def execute(self, sql: str, parameters: Any = ...) -> Any: ...


def _backtick(ident: str) -> str:
    return f"`{ident}`"


def _qualified(schema: str, table: str) -> str:
    return f"{_backtick(schema)}.{_backtick(table)}"


def databricks_create_registry_table_sql(schema: str, table: str) -> str:
    """``CREATE OR REPLACE TABLE ... USING DELTA`` — a full idempotent replace."""
    cols = ",\n  ".join(
        f"{_backtick(c)} {_DATABRICKS_TYPES[c]}" for c in FROZEN_COLUMNS
    )
    return (
        f"CREATE OR REPLACE TABLE {_qualified(schema, table)} "
        f"(\n  {cols}\n) USING DELTA"
    )


def _literal(column: str, value: Any) -> str:
    if column == "entity_id":
        return str(int(value))
    return "'" + str(value).replace("'", "''") + "'"


def _value_tuple(assignment: IdentityAssignment) -> str:
    values = ", ".join(
        _literal(c, getattr(assignment, c)) for c in FROZEN_COLUMNS
    )
    return f"({values})"


def databricks_insert_values_sql(
    schema: str, table: str, batch: Sequence[IdentityAssignment]
) -> str:
    """One ``INSERT INTO ... VALUES`` for a batch of registry rows."""
    cols = ", ".join(_backtick(c) for c in FROZEN_COLUMNS)
    rows = ",\n  ".join(_value_tuple(a) for a in batch)
    return f"INSERT INTO {_qualified(schema, table)} ({cols}) VALUES\n  {rows}"


def bridge_identity_registry(
    cursor: _Cursor,
    assignments: Sequence[IdentityAssignment],
    *,
    schema: str = DEFAULT_SCHEMA,
    table: str = DEFAULT_TABLE,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> int:
    """Mirror the whole registry into a Databricks Delta table; return rows written."""
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS `{schema}`")
    cursor.execute(databricks_create_registry_table_sql(schema, table))
    for start in range(0, len(assignments), batch_rows):
        batch = assignments[start : start + batch_rows]
        cursor.execute(databricks_insert_values_sql(schema, table, batch))
    return len(assignments)
