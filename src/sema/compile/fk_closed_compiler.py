"""S1-06 — FK-closed multi-table compiler: ordered parent→child write (generic).

Materializes an FK-target parent (e.g. omop.person) and an FK-closed child (e.g.
omop.condition-occurrence) so that every child FK references a real parent row.

The guarantee on a warehouse with NO multi-table transaction (Databricks) is an
ORDERING one, not cross-table atomicity: the parent is rebuilt (and swapped in)
BEFORE the child references it, so a child row can never point at a missing
parent. Each single-table write is individually atomic; the child swap is scoped
per study (per source schema) so a multi-study warehouse never clobbers or
orphans another study's rows. FK validity is asserted at rest after both writes.

R29-scanned generic spine: physical names arrive via specs from the policy
boundary; nothing here names ``person``/``condition-occurrence``/OMOP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from sema.compile.compiler_utils import StagingDecision
from sema.compile.fk_closed_compiler_utils import (
    ChildSourceSpec,
    ChildTableSpec,
    ParentTableSpec,
    RegistryJoinSpec,
    build_child_select,
    count_child_scope_sql,
    create_child_table_sql,
    delete_child_scope_sql,
    insert_child_from_temp_sql,
    missing_key_count_sql,
    orphan_fk_count_sql,
    replace_parent_sql,
)

_TEMP_TABLE = "_sema_fk_child_build"


class FkClosureViolation(RuntimeError):
    """Raised when a child FK references no parent row (should be impossible)."""


@dataclass(frozen=True)
class FkClosedResult:
    """The row accounting from one ordered parent→child materialization."""

    parent_rows: int
    child_rows: int
    missing_key_rows: int


class FkClosedCompiler:
    """Ordered, per-study, FK-closed parent→child materialization on DuckDB."""

    def __init__(self, *, no_map_default: int, dialect: str = "duckdb") -> None:
        self._no_map_default = no_map_default
        self._dialect = dialect

    def materialize(
        self,
        conn: Any,
        *,
        parent: ParentTableSpec,
        child: ChildTableSpec,
        source: ChildSourceSpec,
        registry: RegistryJoinSpec,
        decisions: Sequence[StagingDecision],
    ) -> FkClosedResult:
        # Parent FIRST: it must contain every canonical id the child will
        # reference before the child is swapped in (the ordering guarantee).
        self._replace_parent(conn, parent, registry)
        child_rows = self._swap_child(conn, child, source, registry, decisions)
        self._assert_fk_closed(conn, parent, child)
        return FkClosedResult(
            parent_rows=self._scalar(conn, _count_all_sql(parent)),
            child_rows=child_rows,
            missing_key_rows=self._scalar(conn, missing_key_count_sql(source)),
        )

    def _replace_parent(
        self, conn: Any, parent: ParentTableSpec, registry: RegistryJoinSpec
    ) -> None:
        conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{parent.schema}"')
        conn.execute(replace_parent_sql(parent, registry))

    def _swap_child(
        self,
        conn: Any,
        child: ChildTableSpec,
        source: ChildSourceSpec,
        registry: RegistryJoinSpec,
        decisions: Sequence[StagingDecision],
    ) -> int:
        conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{child.schema}"')
        conn.execute(create_child_table_sql(child))
        select = build_child_select(
            child,
            source,
            registry,
            decisions,
            no_map_default=self._no_map_default,
            dialect=self._dialect,
        )
        conn.execute(
            f"CREATE OR REPLACE TEMP TABLE {_TEMP_TABLE} AS "
            f"{select.sql(dialect=self._dialect)}"
        )
        conn.execute("BEGIN TRANSACTION")
        conn.execute(delete_child_scope_sql(child), [source.schema])
        conn.execute(insert_child_from_temp_sql(child, _TEMP_TABLE))
        conn.execute("COMMIT")
        conn.execute(f"DROP TABLE IF EXISTS {_TEMP_TABLE}")
        return self._scalar(conn, count_child_scope_sql(child), [source.schema])

    def _assert_fk_closed(
        self, conn: Any, parent: ParentTableSpec, child: ChildTableSpec
    ) -> None:
        orphans = self._scalar(conn, orphan_fk_count_sql(parent, child))
        if orphans:
            raise FkClosureViolation(
                f"{orphans} rows in {child.schema}.{child.table} reference a "
                f"{parent.schema}.{parent.table} row that does not exist"
            )

    @staticmethod
    def _scalar(conn: Any, sql: str, params: Sequence[Any] | None = None) -> int:
        row = conn.execute(sql, params).fetchone() if params else conn.execute(sql).fetchone()
        return int(row[0]) if row else 0


def _count_all_sql(parent: ParentTableSpec) -> str:
    return f'SELECT COUNT(*) FROM "{parent.schema}"."{parent.table}"'
