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

import sqlglot.expressions as exp

from sema.compile.compiler_utils import (
    CompileContext,
    SourceTableSpec,
    StagingColumns,
    StagingDecision,
    build_staging_select,
)
from sema.compile.staging_backend import (
    DUCKDB_BACKEND,
    StagingBackend,
    StagingCursor,
)
from sema.models.planner._enums import TargetArtifactKind
from sema.models.planner.mapping_plan import MappingPlan
from sema.models.planner.patterns import MappingPattern

__all__ = ["CompiledTransform", "TransformCompiler", "UnsupportedTransformError"]

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
        conn: StagingCursor,
        compiled: CompiledTransform,
        *,
        columns: StagingColumns,
        source: SourceTableSpec,
        staging_schema: str,
        staging_table: str,
        backend: StagingBackend = DUCKDB_BACKEND,
    ) -> int:
        return backend.write_staging(
            conn,
            compiled.select,
            columns=columns,
            source=source,
            staging_schema=staging_schema,
            staging_table=staging_table,
        )


def _plan_pattern(plan: MappingPlan) -> MappingPattern:
    for field_map in plan.field_maps:
        if field_map.pattern is MappingPattern.VOCAB_LOOKUP:
            return field_map.pattern
    raise UnsupportedTransformError("plan has no VOCAB_LOOKUP field map to compile")
