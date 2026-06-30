"""Transform compiler: MappingPlan -> SQLGlot AST -> staging write (US-010).

R29-scanned generic spine: it names no showcase literal. The compiler dispatches
on ``(MappingPattern, TargetArtifactKind)``; Slice 0 implements
``VOCAB_LOOKUP -> TABLE_ROW`` by inlining the resolved decisions as a
``JOIN (VALUES ...)`` (built once, rendered per dialect) and writing the §1.5(b)
staging table with an idempotent temp-build + scoped-swap (atomic replace scoped
on the source schema/table — re-running a study reproduces identical rows).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import duckdb
import sqlglot.expressions as exp

from sema.compile.compiler_utils import (
    CompileContext,
    SourceTableSpec,
    StagingColumns,
    StagingDecision,
    build_staging_select,
    count_scope_sql,
    create_staging_table_sql,
    delete_scope_sql,
    insert_from_temp_sql,
)
from sema.models.planner._enums import TargetArtifactKind
from sema.models.planner.mapping_plan import MappingPlan
from sema.models.planner.patterns import MappingPattern

__all__ = ["CompiledTransform", "TransformCompiler", "UnsupportedTransformError"]

_TEMP_TABLE = "_sema_staging_build"

_Builder = Callable[
    [StagingColumns, SourceTableSpec, CompileContext, Sequence[StagingDecision]],
    exp.Select,
]


class UnsupportedTransformError(ValueError):
    """Raised when no builder is registered for a (pattern, artifact-kind) pair."""


@dataclass(frozen=True)
class CompiledTransform:
    """A dialect-agnostic compiled staging SELECT."""

    select: exp.Select

    def sql(self, dialect: str) -> str:
        return self.select.sql(dialect=dialect)


class TransformCompiler:
    """Compile a MappingPlan into a staging write and execute it on DuckDB."""

    def __init__(self) -> None:
        self._builders: dict[tuple[MappingPattern, TargetArtifactKind], _Builder] = {
            (MappingPattern.VOCAB_LOOKUP, TargetArtifactKind.TABLE_ROW): (
                build_staging_select
            ),
        }

    def compile(
        self,
        plan: MappingPlan,
        columns: StagingColumns,
        source: SourceTableSpec,
        context: CompileContext,
        decisions: Sequence[StagingDecision],
        *,
        artifact_kind: TargetArtifactKind = TargetArtifactKind.TABLE_ROW,
    ) -> CompiledTransform:
        pattern = _plan_pattern(plan)
        builder = self._builders.get((pattern, artifact_kind))
        if builder is None:
            raise UnsupportedTransformError(
                f"no compiler for ({pattern.value}, {artifact_kind.value})"
            )
        return CompiledTransform(select=builder(columns, source, context, decisions))

    def execute(
        self,
        conn: duckdb.DuckDBPyConnection,
        compiled: CompiledTransform,
        *,
        columns: StagingColumns,
        source: SourceTableSpec,
        staging_schema: str,
        staging_table: str,
    ) -> int:
        conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{staging_schema}"')
        conn.execute(create_staging_table_sql(columns, staging_schema, staging_table))
        conn.execute(
            f"CREATE OR REPLACE TEMP TABLE {_TEMP_TABLE} AS {compiled.sql('duckdb')}"
        )
        scope = [source.schema, source.table]
        conn.execute("BEGIN TRANSACTION")
        conn.execute(delete_scope_sql(columns, staging_schema, staging_table), scope)
        conn.execute(
            insert_from_temp_sql(columns, staging_schema, staging_table, _TEMP_TABLE)
        )
        conn.execute("COMMIT")
        conn.execute(f"DROP TABLE IF EXISTS {_TEMP_TABLE}")
        row = conn.execute(
            count_scope_sql(columns, staging_schema, staging_table), scope
        ).fetchone()
        return int(row[0]) if row else 0


def _plan_pattern(plan: MappingPlan) -> MappingPattern:
    for field_map in plan.field_maps:
        if field_map.pattern is MappingPattern.VOCAB_LOOKUP:
            return field_map.pattern
    raise UnsupportedTransformError("plan has no VOCAB_LOOKUP field map to compile")
