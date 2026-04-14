"""Runtime telemetry: per-table and aggregate pipeline metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sema.models.stages import (
    StageBResult,
    StageAResult,
)


@dataclass(frozen=True)
class TableTelemetry:
    """Per-table telemetry extracted from stage outputs."""

    table_ref: str
    stage_a_calls: int
    stage_b_batches_attempted: int
    stage_b_batches_succeeded: int
    stage_c_calls: int
    b_outcome: str
    retries_used: int
    splits_used: int
    rescues_used: int
    raw_coverage_pct: float
    critical_coverage_pct: float
    c_columns_flagged: int
    total_columns: int
    stage_a_latency_ms: int
    stage_b_latency_ms: int
    stage_c_latency_ms: int
    tokens_input: int
    tokens_output: int

    @property
    def c_trigger_rate(self) -> float:
        if self.total_columns == 0:
            return 0.0
        return self.c_columns_flagged / self.total_columns

    @property
    def total_latency_ms(self) -> int:
        return (
            self.stage_a_latency_ms
            + self.stage_b_latency_ms
            + self.stage_c_latency_ms
        )

    @classmethod
    def from_stages(
        cls,
        table_ref: str,
        stage_a: StageAResult,
        stage_b: StageBResult,
        *,
        stage_a_latency_ms: int = 0,
        stage_b_latency_ms: int = 0,
        stage_c_latency_ms: int = 0,
        stage_c_calls: int = 0,
        tokens_input: int = 0,
        tokens_output: int = 0,
    ) -> TableTelemetry:
        all_cols = [
            col
            for br in stage_b.batch_results
            for col in br.columns
        ]
        c_flagged = sum(1 for c in all_cols if c.needs_stage_c)
        total = stage_b.raw_coverage.total

        return cls(
            table_ref=table_ref,
            stage_a_calls=1,
            stage_b_batches_attempted=len(stage_b.batch_results),
            stage_b_batches_succeeded=len(stage_b.batch_results),
            stage_c_calls=stage_c_calls,
            b_outcome=stage_b.status,
            retries_used=stage_b.retries_used,
            splits_used=stage_b.splits_used,
            rescues_used=stage_b.rescues_used,
            raw_coverage_pct=stage_b.raw_coverage.pct,
            critical_coverage_pct=stage_b.critical_coverage.pct,
            c_columns_flagged=c_flagged,
            total_columns=total,
            stage_a_latency_ms=stage_a_latency_ms,
            stage_b_latency_ms=stage_b_latency_ms,
            stage_c_latency_ms=stage_c_latency_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
        )


@dataclass(frozen=True)
class PipelineTelemetry:
    """Aggregate telemetry across all tables in a run."""

    table_count: int
    b_success_count: int
    b_partial_count: int
    b_failed_count: int
    avg_raw_coverage_pct: float
    avg_latency_ms: float
    avg_c_trigger_rate: float
    total_retries: int
    total_splits: int
    total_rescues: int
    total_tokens_input: int
    total_tokens_output: int

    @classmethod
    def aggregate(
        cls, tables: list[TableTelemetry],
    ) -> PipelineTelemetry:
        if not tables:
            return cls(
                table_count=0,
                b_success_count=0,
                b_partial_count=0,
                b_failed_count=0,
                avg_raw_coverage_pct=0.0,
                avg_latency_ms=0.0,
                avg_c_trigger_rate=0.0,
                total_retries=0,
                total_splits=0,
                total_rescues=0,
                total_tokens_input=0,
                total_tokens_output=0,
            )

        n = len(tables)
        return cls(
            table_count=n,
            b_success_count=sum(
                1 for t in tables if t.b_outcome == "B_SUCCESS"
            ),
            b_partial_count=sum(
                1 for t in tables if t.b_outcome == "B_PARTIAL"
            ),
            b_failed_count=sum(
                1 for t in tables if t.b_outcome == "B_FAILED"
            ),
            avg_raw_coverage_pct=sum(
                t.raw_coverage_pct for t in tables
            ) / n,
            avg_latency_ms=sum(
                t.total_latency_ms for t in tables
            ) / n,
            avg_c_trigger_rate=sum(
                t.c_trigger_rate for t in tables
            ) / n,
            total_retries=sum(t.retries_used for t in tables),
            total_splits=sum(t.splits_used for t in tables),
            total_rescues=sum(t.rescues_used for t in tables),
            total_tokens_input=sum(
                t.tokens_input for t in tables
            ),
            total_tokens_output=sum(
                t.tokens_output for t in tables
            ),
        )


def build_milestone_report(
    label: str,
    telemetry: PipelineTelemetry,
    diff_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a milestone evaluation report."""
    report: dict[str, Any] = {
        "label": label,
        "telemetry": {
            "table_count": telemetry.table_count,
            "b_outcome_distribution": {
                "success": telemetry.b_success_count,
                "partial": telemetry.b_partial_count,
                "failed": telemetry.b_failed_count,
            },
            "avg_raw_coverage_pct": round(
                telemetry.avg_raw_coverage_pct, 4,
            ),
            "avg_latency_ms": round(telemetry.avg_latency_ms, 1),
            "avg_c_trigger_rate": round(
                telemetry.avg_c_trigger_rate, 4,
            ),
            "recovery": {
                "total_retries": telemetry.total_retries,
                "total_splits": telemetry.total_splits,
                "total_rescues": telemetry.total_rescues,
            },
            "tokens": {
                "input": telemetry.total_tokens_input,
                "output": telemetry.total_tokens_output,
            },
        },
    }
    if diff_summary:
        report["semantic_churn"] = {
            "added": diff_summary.get("added_count", 0),
            "removed": diff_summary.get("removed_count", 0),
            "changed": diff_summary.get("changed_count", 0),
            "total_before": diff_summary.get("total_before", 0),
            "total_after": diff_summary.get("total_after", 0),
        }
    return report
