"""Tests for runtime telemetry instrumentation (tasks 6.1-6.7)."""
import pytest

from sema.models.stages import (
    StageBCoverage,
    StageBResult,
    StageBBatchResult,
    StageBColumnResult,
    StageAResult,
    StageStatus,
    UnresolvedColumn,
)

pytestmark = pytest.mark.unit


def _make_stage_a() -> StageAResult:
    return StageAResult(
        primary_entity="Patient",
        grain_hypothesis="one row per patient",
        confidence=0.9,
    )


def _make_b_column(
    name: str,
    needs_c: bool = False,
) -> StageBColumnResult:
    return StageBColumnResult(
        column=name,
        canonical_property_label=name,
        semantic_type="identifier",
        confidence=0.9,
        needs_stage_c=needs_c,
    )


def _make_stage_b_success(
    cols: list[str],
    needs_c_cols: list[str] | None = None,
) -> StageBResult:
    needs_c = needs_c_cols or []
    columns = [_make_b_column(c, c in needs_c) for c in cols]
    batch = StageBBatchResult(columns=columns)
    return StageBResult(
        status="B_SUCCESS",
        batch_results=[batch],
        raw_coverage=StageBCoverage(
            classified=len(cols), total=len(cols), pct=1.0,
        ),
        critical_coverage=StageBCoverage(
            classified=len(cols), total=len(cols), pct=1.0,
        ),
    )


def _make_stage_b_partial(
    classified: list[str],
    total: int,
    unresolved: list[str] | None = None,
) -> StageBResult:
    columns = [_make_b_column(c) for c in classified]
    batch = StageBBatchResult(columns=columns)
    unresolved_cols = [
        UnresolvedColumn(
            column=c, reason="execution_failure", tier="peripheral",
        )
        for c in (unresolved or [])
    ]
    return StageBResult(
        status="B_PARTIAL",
        batch_results=[batch],
        raw_coverage=StageBCoverage(
            classified=len(classified),
            total=total,
            pct=len(classified) / total if total else 0.0,
        ),
        critical_coverage=StageBCoverage(
            classified=len(classified), total=total, pct=1.0,
        ),
        unresolved_columns=unresolved_cols,
        retries_used=1,
        splits_used=1,
        rescues_used=0,
    )


class TestTableTelemetry:
    """Per-table telemetry from stage outputs."""

    def test_from_staged_output(self) -> None:
        from sema.eval.telemetry import TableTelemetry

        stage_a = _make_stage_a()
        stage_b = _make_stage_b_success(["col1", "col2", "col3"])
        tel = TableTelemetry.from_stages(
            table_ref="unity://cat.sch.patient",
            stage_a=stage_a,
            stage_b=stage_b,
        )
        assert tel.table_ref == "unity://cat.sch.patient"
        assert tel.stage_a_calls == 1
        assert tel.stage_b_batches_attempted == 1
        assert tel.stage_b_batches_succeeded == 1
        assert tel.stage_c_calls == 0
        assert tel.b_outcome == "B_SUCCESS"

    def test_b_recovery_metrics(self) -> None:
        from sema.eval.telemetry import TableTelemetry

        stage_b = _make_stage_b_partial(
            ["a", "b", "c"], 4, unresolved=["d"],
        )
        tel = TableTelemetry.from_stages(
            table_ref="t",
            stage_a=_make_stage_a(),
            stage_b=stage_b,
        )
        assert tel.retries_used == 1
        assert tel.splits_used == 1
        assert tel.rescues_used == 0
        assert tel.raw_coverage_pct == 0.75
        assert tel.critical_coverage_pct == 1.0
        assert tel.b_outcome == "B_PARTIAL"

    def test_c_trigger_rate(self) -> None:
        from sema.eval.telemetry import TableTelemetry

        stage_b = _make_stage_b_success(
            ["col1", "col2", "col3", "col4"],
            needs_c_cols=["col2", "col4"],
        )
        tel = TableTelemetry.from_stages(
            table_ref="t",
            stage_a=_make_stage_a(),
            stage_b=stage_b,
        )
        assert tel.c_columns_flagged == 2
        assert tel.total_columns == 4
        assert tel.c_trigger_rate == 0.5

    def test_latency_tracking(self) -> None:
        from sema.eval.telemetry import TableTelemetry

        stage_b = _make_stage_b_success(["col1"])
        tel = TableTelemetry.from_stages(
            table_ref="t",
            stage_a=_make_stage_a(),
            stage_b=stage_b,
            stage_a_latency_ms=150,
            stage_b_latency_ms=800,
        )
        assert tel.stage_a_latency_ms == 150
        assert tel.stage_b_latency_ms == 800
        assert tel.total_latency_ms == 950

    def test_token_usage_defaults_zero(self) -> None:
        from sema.eval.telemetry import TableTelemetry

        stage_b = _make_stage_b_success(["col1"])
        tel = TableTelemetry.from_stages(
            table_ref="t",
            stage_a=_make_stage_a(),
            stage_b=stage_b,
        )
        assert tel.tokens_input == 0
        assert tel.tokens_output == 0


class TestPipelineTelemetry:
    """Aggregate telemetry across tables."""

    def test_aggregate_empty(self) -> None:
        from sema.eval.telemetry import PipelineTelemetry

        agg = PipelineTelemetry.aggregate([])
        assert agg.table_count == 0
        assert agg.avg_latency_ms == 0.0

    def test_aggregate_b_outcome_distribution(self) -> None:
        from sema.eval.telemetry import (
            PipelineTelemetry,
            TableTelemetry,
        )

        t1 = TableTelemetry.from_stages(
            "t1", _make_stage_a(),
            _make_stage_b_success(["a", "b"]),
        )
        t2 = TableTelemetry.from_stages(
            "t2", _make_stage_a(),
            _make_stage_b_partial(["a"], 2, ["b"]),
        )
        t3 = TableTelemetry.from_stages(
            "t3", _make_stage_a(),
            _make_stage_b_success(["x"]),
        )
        agg = PipelineTelemetry.aggregate([t1, t2, t3])
        assert agg.b_success_count == 2
        assert agg.b_partial_count == 1
        assert agg.b_failed_count == 0

    def test_aggregate_avg_coverage(self) -> None:
        from sema.eval.telemetry import (
            PipelineTelemetry,
            TableTelemetry,
        )

        t1 = TableTelemetry.from_stages(
            "t1", _make_stage_a(),
            _make_stage_b_success(["a", "b", "c", "d"]),
        )
        t2 = TableTelemetry.from_stages(
            "t2", _make_stage_a(),
            _make_stage_b_partial(["a", "b"], 4, ["c", "d"]),
        )
        agg = PipelineTelemetry.aggregate([t1, t2])
        assert agg.avg_raw_coverage_pct == pytest.approx(0.75)

    def test_aggregate_latency(self) -> None:
        from sema.eval.telemetry import (
            PipelineTelemetry,
            TableTelemetry,
        )

        t1 = TableTelemetry.from_stages(
            "t1", _make_stage_a(),
            _make_stage_b_success(["a"]),
            stage_a_latency_ms=100, stage_b_latency_ms=200,
        )
        t2 = TableTelemetry.from_stages(
            "t2", _make_stage_a(),
            _make_stage_b_success(["b"]),
            stage_a_latency_ms=300, stage_b_latency_ms=400,
        )
        agg = PipelineTelemetry.aggregate([t1, t2])
        assert agg.avg_latency_ms == pytest.approx(500.0)

    def test_aggregate_c_trigger_rate(self) -> None:
        from sema.eval.telemetry import (
            PipelineTelemetry,
            TableTelemetry,
        )

        t1 = TableTelemetry.from_stages(
            "t1", _make_stage_a(),
            _make_stage_b_success(
                ["a", "b", "c", "d"], needs_c_cols=["a", "b"],
            ),
        )
        t2 = TableTelemetry.from_stages(
            "t2", _make_stage_a(),
            _make_stage_b_success(
                ["x", "y", "z", "w"], needs_c_cols=[],
            ),
        )
        agg = PipelineTelemetry.aggregate([t1, t2])
        assert agg.avg_c_trigger_rate == pytest.approx(0.25)

    def test_aggregate_recovery_totals(self) -> None:
        from sema.eval.telemetry import (
            PipelineTelemetry,
            TableTelemetry,
        )

        t1 = TableTelemetry.from_stages(
            "t1", _make_stage_a(),
            _make_stage_b_partial(["a"], 2, ["b"]),
        )
        t2 = TableTelemetry.from_stages(
            "t2", _make_stage_a(),
            _make_stage_b_partial(["x"], 2, ["y"]),
        )
        agg = PipelineTelemetry.aggregate([t1, t2])
        assert agg.total_retries == 2
        assert agg.total_splits == 2


class TestTelemetryReport:
    """Report generation combining telemetry with diff stats."""

    def test_report_structure(self) -> None:
        from sema.eval.telemetry import (
            PipelineTelemetry,
            TableTelemetry,
            build_milestone_report,
        )

        t1 = TableTelemetry.from_stages(
            "t1", _make_stage_a(),
            _make_stage_b_success(["a", "b"]),
            stage_a_latency_ms=100, stage_b_latency_ms=200,
        )
        agg = PipelineTelemetry.aggregate([t1])
        diff_summary = {
            "added_count": 3,
            "removed_count": 1,
            "changed_count": 2,
            "total_before": 10,
            "total_after": 12,
        }
        report = build_milestone_report(
            label="step2_baseline",
            telemetry=agg,
            diff_summary=diff_summary,
        )
        assert report["label"] == "step2_baseline"
        assert "telemetry" in report
        assert "semantic_churn" in report
        assert report["semantic_churn"]["added"] == 3
        assert report["semantic_churn"]["removed"] == 1
