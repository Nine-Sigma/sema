"""Gate D-lite staging QA — DuckDB reader + orchestration (US-011).

Reads the §1.5(b) staging table written by the US-010 compiler (physical column
names supplied by the US-004 policy via :class:`StagingColumns`, so this module
names no showcase literal) and runs the three checks in
:mod:`sema.eval.staging_qa_utils`. A bad staging table fails the report with a
structured reason before it is called "done".
"""

from __future__ import annotations

from typing import Any, Protocol

from sema.compile.compiler_utils import StagingColumns
from sema.eval.mapping_goldset import GoldSet
from sema.eval.staging_qa_utils import (
    StagingQAReport,
    StagingRow,
    check_no_map_accounting,
    check_null_rate,
    check_row_count,
)

__all__ = [
    "read_staging_rows",
    "run_staging_qa",
    "staging_scope_count",
]


class _Conn(Protocol):
    def execute(self, sql: str, params: list[Any] = ...) -> Any: ...


def _qualified(schema: str, table: str) -> str:
    return f'"{schema}"."{table}"'


def read_staging_rows(
    conn: _Conn,
    columns: StagingColumns,
    staging_schema: str,
    staging_table: str,
    source_schema: str,
    source_table: str,
) -> list[StagingRow]:
    """Read the staged rows for one source scope into generic QA rows."""
    sql = (
        f'SELECT "{columns.source_value_column}", '
        f'"{columns.target_concept_column}", "{columns.resolution_status}" '
        f"FROM {_qualified(staging_schema, staging_table)} "
        f'WHERE "{columns.source_schema}" = ? AND "{columns.source_table}" = ?'
    )
    result = conn.execute(sql, [source_schema, source_table]).fetchall()
    return [
        StagingRow(
            source_value=str(row[0]),
            target_value=None if row[1] is None else int(row[1]),
            resolution_status=str(row[2]),
        )
        for row in result
    ]


def staging_scope_count(
    conn: _Conn,
    columns: StagingColumns,
    staging_schema: str,
    staging_table: str,
    source_schema: str,
    source_table: str,
) -> int:
    """Count staged rows for one source scope."""
    sql = (
        f"SELECT COUNT(*) FROM {_qualified(staging_schema, staging_table)} "
        f'WHERE "{columns.source_schema}" = ? AND "{columns.source_table}" = ?'
    )
    row = conn.execute(sql, [source_schema, source_table]).fetchone()
    return int(row[0]) if row else 0


def run_staging_qa(
    conn: _Conn,
    *,
    columns: StagingColumns,
    staging_schema: str,
    staging_table: str,
    source_schema: str,
    source_table: str,
    expected_row_count: int,
    gold_set: GoldSet,
) -> StagingQAReport:
    """Run the three Gate D-lite checks over one staged source scope."""
    actual = staging_scope_count(
        conn, columns, staging_schema, staging_table, source_schema, source_table
    )
    rows = read_staging_rows(
        conn, columns, staging_schema, staging_table, source_schema, source_table
    )
    return StagingQAReport(
        checks=(
            check_row_count(actual, expected_row_count),
            check_null_rate(rows),
            check_no_map_accounting(rows, gold_set),
        )
    )
