"""Gate D-lite staging QA — DuckDB reader + orchestration (US-011).

Reads the §1.5(b) staging table written by the US-010 compiler (physical column
names supplied by the US-004 policy via :class:`StagingColumns`, so this module
names no showcase literal) and runs the three checks in
:mod:`sema.eval.staging_qa_utils`. A bad staging table fails the report with a
structured reason before it is called "done".
"""

from __future__ import annotations

from typing import Any, Sequence

from sema.compile.compiler_utils import StagingColumns
from sema.compile.fk_closed_compiler_utils import (
    ChildSourceSpec,
    ChildTableSpec,
    ParentTableSpec,
    count_child_scope_sql,
    missing_key_count_sql,
    orphan_fk_count_sql,
)
from sema.compile.staging_backend import (
    DUCKDB_BACKEND,
    StagingBackend,
    StagingCursor,
)
from sema.eval.mapping_goldset import GoldSet
from sema.eval.staging_qa_utils import (
    StagingQAReport,
    StagingRow,
    check_fk_closure,
    check_missing_key_disposition,
    check_no_map_accounting,
    check_null_rate,
    check_required_not_null,
    check_row_count,
)

__all__ = [
    "count_column_nulls",
    "read_staging_rows",
    "run_fk_closed_qa",
    "run_staging_qa",
    "staging_scope_count",
]


def _scalar(conn: Any, sql: str, params: Sequence[Any] | None = None) -> int:
    row = conn.execute(sql, params).fetchone() if params else conn.execute(sql).fetchone()
    return int(row[0]) if row else 0


def count_column_nulls(
    conn: Any,
    schema: str,
    table: str,
    columns: Sequence[str],
    *,
    scope_column: str | None = None,
    scope_value: str | None = None,
) -> dict[str, int]:
    """Per-column NULL counts (optionally scoped to one partition)."""
    if not columns:
        return {}
    sums = ", ".join(
        f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END)' for c in columns
    )
    sql = f'SELECT {sums} FROM "{schema}"."{table}"'
    params: list[Any] | None = None
    if scope_column is not None:
        sql += f' WHERE "{scope_column}" = ?'
        params = [scope_value]
    row = conn.execute(sql, params).fetchone() if params else conn.execute(sql).fetchone()
    return {c: int(row[i] or 0) for i, c in enumerate(columns)}


def run_fk_closed_qa(
    conn: Any,
    *,
    parent: ParentTableSpec,
    child: ChildTableSpec,
    source: ChildSourceSpec,
    required_fields: Sequence[str],
    source_row_count: int,
) -> StagingQAReport:
    """Gate-D-lite over the FK-closed shape (S1-07): closure + required-null +
    missing-key accounting, scoped to one study."""
    orphans = _scalar(conn, orphan_fk_count_sql(parent, child))
    missing = _scalar(conn, missing_key_count_sql(source))
    written = _scalar(conn, count_child_scope_sql(child), [source.schema])
    null_counts = count_column_nulls(
        conn,
        child.schema,
        child.table,
        required_fields,
        scope_column=child.scope_schema_column,
        scope_value=source.schema,
    )
    return StagingQAReport(
        checks=(
            check_fk_closure(orphans),
            check_required_not_null(null_counts),
            check_missing_key_disposition(
                written_rows=written,
                missing_key_rows=missing,
                source_rows=source_row_count,
            ),
        )
    )


def read_staging_rows(
    conn: StagingCursor,
    columns: StagingColumns,
    staging_schema: str,
    staging_table: str,
    source_schema: str,
    source_table: str,
    *,
    backend: StagingBackend = DUCKDB_BACKEND,
) -> list[StagingRow]:
    """Read the staged rows for one source scope into generic QA rows."""
    predicate, params = backend.scope_predicate(columns, source_schema, source_table)
    select_cols = ", ".join(
        backend.quote(name)
        for name in (
            columns.source_value_column,
            columns.target_concept_column,
            columns.resolution_status,
        )
    )
    sql = (
        f"SELECT {select_cols} "
        f"FROM {backend.qualified(staging_schema, staging_table)} "
        f"WHERE {predicate}"
    )
    result = backend.query(conn, sql, params)
    return [
        StagingRow(
            source_value=str(row[0]),
            target_value=None if row[1] is None else int(row[1]),
            resolution_status=str(row[2]),
        )
        for row in result
    ]


def staging_scope_count(
    conn: StagingCursor,
    columns: StagingColumns,
    staging_schema: str,
    staging_table: str,
    source_schema: str,
    source_table: str,
    *,
    backend: StagingBackend = DUCKDB_BACKEND,
) -> int:
    """Count staged rows for one source scope."""
    predicate, params = backend.scope_predicate(columns, source_schema, source_table)
    sql = (
        f"SELECT COUNT(*) FROM {backend.qualified(staging_schema, staging_table)} "
        f"WHERE {predicate}"
    )
    row = backend.fetch_one(conn, sql, params)
    return int(row[0]) if row else 0


def run_staging_qa(
    conn: StagingCursor,
    *,
    columns: StagingColumns,
    staging_schema: str,
    staging_table: str,
    source_schema: str,
    source_table: str,
    expected_row_count: int,
    gold_set: GoldSet,
    backend: StagingBackend = DUCKDB_BACKEND,
) -> StagingQAReport:
    """Run the three Gate D-lite checks over one staged source scope."""
    actual = staging_scope_count(
        conn, columns, staging_schema, staging_table, source_schema, source_table,
        backend=backend,
    )
    rows = read_staging_rows(
        conn, columns, staging_schema, staging_table, source_schema, source_table,
        backend=backend,
    )
    return StagingQAReport(
        checks=(
            check_row_count(actual, expected_row_count),
            check_null_rate(rows),
            check_no_map_accounting(rows, gold_set),
        )
    )
