"""S1-06/S1-08 — FK-closed multi-table compiler: ordered parent→child write.

Materializes an FK-target parent (e.g. omop.person) and an FK-closed child (e.g.
omop.condition-occurrence) so that every child FK references a real parent row.

The guarantee on a warehouse with NO multi-table transaction (Databricks) is an
ORDERING one, not cross-table atomicity: the parent is rebuilt (and swapped in)
BEFORE the child references it, so a child row can never point at a missing
parent. Each single-table write is individually atomic; the child swap is scoped
per study (per source schema) so a multi-study warehouse never clobbers or
orphans another study's rows. FK validity is asserted at rest after both writes.

The warehouse-specific write mechanics live behind an :class:`FkBackend` (DuckDB
temp-swap vs Databricks Delta ``REPLACE WHERE``); this module stays free of
dialect branches. R29-scanned generic spine: physical names arrive via specs from
the policy boundary; nothing here names ``person``/``condition-occurrence``/OMOP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sema.compile.compiler_utils import StagingDecision
from sema.compile.fk_backend import DUCKDB_FK_BACKEND, FkBackend, FkCursor
from sema.compile.fk_closed_compiler_utils import (
    ChildSourceSpec,
    ChildTableSpec,
    ParentTableSpec,
    RegistryJoinSpec,
    build_child_select,
)

__all__ = ["FkClosedCompiler", "FkClosedResult", "FkClosureViolation"]


class FkClosureViolation(RuntimeError):
    """Raised when a child FK references no parent row (should be impossible)."""


@dataclass(frozen=True)
class FkClosedResult:
    """The row accounting from one ordered parent→child materialization."""

    parent_rows: int
    child_rows: int
    missing_key_rows: int


class FkClosedCompiler:
    """Ordered, per-study, FK-closed parent→child materialization (any backend)."""

    def __init__(
        self, *, no_map_default: int, backend: FkBackend = DUCKDB_FK_BACKEND
    ) -> None:
        self._no_map_default = no_map_default
        self._backend = backend

    def materialize(
        self,
        conn: FkCursor,
        *,
        parent: ParentTableSpec,
        child: ChildTableSpec,
        source: ChildSourceSpec,
        registry: RegistryJoinSpec,
        decisions: Sequence[StagingDecision],
    ) -> FkClosedResult:
        # Parent FIRST: it must contain every canonical id the child will
        # reference before the child is swapped in (the ordering guarantee).
        self._backend.replace_parent(conn, parent, registry)
        select = build_child_select(
            child,
            source,
            registry,
            decisions,
            no_map_default=self._no_map_default,
            dialect=self._backend.dialect,
        )
        child_rows = self._backend.write_child(conn, child, source, select)
        self._assert_fk_closed(conn, parent, child)
        return FkClosedResult(
            parent_rows=self._backend.count_all(conn, parent),
            child_rows=child_rows,
            missing_key_rows=self._backend.missing_key_count(conn, source),
        )

    def rebuild_after_collapse(
        self,
        conn: FkCursor,
        *,
        parent: ParentTableSpec,
        child: ChildTableSpec,
        sources: Sequence[ChildSourceSpec],
        registry: RegistryJoinSpec,
        decisions: Sequence[StagingDecision],
    ) -> FkClosedResult:
        """Rematerialize every child scope through a collapsed registry (Stage B).

        A Stage B collapse RETIRES canonical ids, so the parent SHRINKS — the
        inverse of Stage A. The order flips accordingly: rebuild every child
        scope first (each FK now points at a surviving id) and shrink the parent
        LAST, so no not-yet-rebuilt sibling scope is left referencing a retired
        parent. The child's surrogate PK is preserved (source-derived, S1-05);
        each row's FK is recomputed from the current registry.

        ``sources`` MUST be the COMPLETE, DISTINCT set of studies referencing
        ``parent``. The ``_assert_fk_closed`` guard is not a full check: it
        catches an omitted study ONLY if that study still references a retired
        id — a survivors-only omission, or an empty collapse, leaves stale child
        rows undetected. Duplicate sources cause duplicate rewrites and inflated
        ``child_rows``. The caller owns completeness and distinctness.
        """
        child_rows = 0
        missing = 0
        for source in sources:
            select = build_child_select(
                child,
                source,
                registry,
                decisions,
                no_map_default=self._no_map_default,
                dialect=self._backend.dialect,
            )
            child_rows += self._backend.write_child(conn, child, source, select)
            missing += self._backend.missing_key_count(conn, source)
        self._backend.replace_parent(conn, parent, registry)
        self._assert_fk_closed(conn, parent, child)
        return FkClosedResult(
            parent_rows=self._backend.count_all(conn, parent),
            child_rows=child_rows,
            missing_key_rows=missing,
        )

    def _assert_fk_closed(
        self, conn: FkCursor, parent: ParentTableSpec, child: ChildTableSpec
    ) -> None:
        orphans = self._backend.orphan_fk_count(conn, parent, child)
        if orphans:
            raise FkClosureViolation(
                f"{orphans} rows in {child.schema}.{child.table} reference a "
                f"{parent.schema}.{parent.table} row that does not exist"
            )
