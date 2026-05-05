"""Run-level quality budget (Section 6)."""
from __future__ import annotations

import pytest

from sema.models.config import BuildConfig
from sema.pipeline.build import TableResult

pytestmark = pytest.mark.unit


def _failed(table_ref: str, stage: str, reason: str = "fail",
            tier: str | None = None) -> TableResult:
    return TableResult.failed(table_ref, stage, reason, metadata_tier=tier)


def _success(table_ref: str, partial: bool = False,
             tier: str | None = None) -> TableResult:
    return TableResult.success(
        table_ref, entities=1, properties=1,
        partial=partial, metadata_tier=tier,
    )


def _skipped(table_ref: str, reason: str) -> TableResult:
    return TableResult.skipped(table_ref, reason)


def _cfg(**overrides) -> BuildConfig:
    base = BuildConfig()
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


class TestComputeBreakdown:
    def test_separates_skipped_resume_from_skipped_other(self):
        from sema.pipeline.quality_budget import compute_breakdown

        results = [
            _success("t1"),
            _skipped("t2", "resume: assertions exist"),
            _skipped("t3", "no table metadata"),
        ]
        b = compute_breakdown(results)
        assert b.succeeded == 1
        assert b.skipped_resume == 1
        assert b.skipped_other == 1

    def test_classifies_failure_categories(self):
        from sema.pipeline.quality_budget import compute_breakdown

        results = [
            _failed("t1", "L2 stage_b"),
            _failed("t2", "L2 stage_a"),
            _failed("t3", "circuit_breaker"),
            _failed("t4", "unknown"),
        ]
        b = compute_breakdown(results)
        assert b.stage_b_failed == 1
        assert b.stage_a_failed == 1
        assert b.circuit_open == 1
        assert b.other_failed == 1


class TestStageBBudget:
    def test_40pct_b_failed_triggers(self):
        from sema.pipeline.quality_budget import (
            QualityBudgetExceeded, enforce_quality_budget,
        )

        # 12 b_failed / 30 graph-contributing = 0.40
        results = (
            [_success(f"s{i}") for i in range(18)]
            + [_failed(f"f{i}", "L2 stage_b") for i in range(12)]
        )
        with pytest.raises(QualityBudgetExceeded) as exc_info:
            enforce_quality_budget(results, _cfg())

        err = exc_info.value
        assert err.trigger == "stage_b_failure_rate"

    def test_20pct_b_failed_does_not_trigger(self):
        from sema.pipeline.quality_budget import enforce_quality_budget

        results = (
            [_success(f"s{i}") for i in range(24)]
            + [_failed(f"f{i}", "L2 stage_b") for i in range(6)]
        )
        # No raise
        enforce_quality_budget(results, _cfg())

    def test_tiny_run_below_min_processed(self):
        from sema.pipeline.quality_budget import enforce_quality_budget

        results = [
            _success("s1"),
            _failed("f1", "L2 stage_b"),
            _failed("f2", "L2 stage_b"),
        ]
        # Under 5-table min — no raise even at 67% rate
        enforce_quality_budget(results, _cfg())

    def test_circuit_open_excluded_from_trigger1(self):
        from sema.pipeline.quality_budget import enforce_quality_budget

        # 4 b_failed / 30 graph-contributing = 0.133, no trigger 1
        # But 3 circuit_open + 4 stage_b_failed → 7/30 = 0.233, no T2
        results = (
            [_success(f"s{i}") for i in range(23)]
            + [_failed(f"f{i}", "L2 stage_b") for i in range(4)]
            + [_failed(f"co{i}", "circuit_breaker") for i in range(3)]
        )
        enforce_quality_budget(results, _cfg())

    def test_resume_retry_b_failed_within_budget(self):
        """Resume of 8 against 29 prior, all 8 hit Stage B fail.
        T1 denominator counts 29 skipped_resume → rate = 8/37 = 0.216
        which stays under 0.30. T2 denominator = 8 (excludes resume),
        non_contributing numerator = 0 (stage_b is graph-contributing)
        → rate = 0. Run continues."""
        from sema.pipeline.quality_budget import enforce_quality_budget

        results = (
            [_failed(f"f{i}", "L2 stage_b") for i in range(8)]
            + [_skipped(f"s{i}", "resume: ok") for i in range(29)]
        )
        # Should NOT raise
        enforce_quality_budget(results, _cfg())


class TestRunReliabilityBudget:
    def test_resume_retry_all_stage_a_storm(self):
        """All 8 retries hit stage_a — trigger 2 fires."""
        from sema.pipeline.quality_budget import (
            QualityBudgetExceeded, enforce_quality_budget,
        )

        # 8 stage_a_failed, 29 skipped_resume
        # T1: 0 / (0 + 0 + 29) = 0 → no trigger
        # T2: 8 / (0 + 8 + 0) = 1.0 → triggers
        results = (
            [_failed(f"f{i}", "L2 stage_a") for i in range(8)]
            + [_skipped(f"s{i}", "resume: ok") for i in range(29)]
        )
        with pytest.raises(QualityBudgetExceeded) as exc_info:
            enforce_quality_budget(results, _cfg())
        assert exc_info.value.trigger == "run_non_contributing_rate"

    def test_healthy_run_stays_under(self):
        from sema.pipeline.quality_budget import enforce_quality_budget

        # 25 success, 3 b_fail, 2 stage_a_fail
        # T1: 3/28 ≈ 0.107 → no
        # T2: 2/30 ≈ 0.067 → no
        results = (
            [_success(f"s{i}") for i in range(25)]
            + [_failed(f"b{i}", "L2 stage_b") for i in range(3)]
            + [_failed(f"a{i}", "L2 stage_a") for i in range(2)]
        )
        enforce_quality_budget(results, _cfg())

    def test_skipped_other_excluded_from_t1_included_in_t2(self):
        from sema.pipeline.quality_budget import compute_breakdown

        results = [
            _success("s1"), _success("s2"), _success("s3"),
            _success("s4"), _success("s5"),
            _skipped("sk1", "no table metadata"),  # skipped_other
        ]
        b = compute_breakdown(results)
        assert b.skipped_other == 1
        assert b.skipped_resume == 0


class TestStableOrdering:
    def test_both_triggers_exceed_t1_fires_first(self):
        from sema.pipeline.quality_budget import (
            QualityBudgetExceeded, enforce_quality_budget,
        )

        # 3 ok, 5 b_fail, 22 circuit_open
        # T1: 5 / (3 + 5 + 0) = 0.625 → exceeds 0.30
        # T2: 22 / (3 + 27 + 0) = 0.733 → exceeds 0.40
        results = (
            [_success(f"s{i}") for i in range(3)]
            + [_failed(f"b{i}", "L2 stage_b") for i in range(5)]
            + [_failed(f"co{i}", "circuit_breaker") for i in range(22)]
        )
        with pytest.raises(QualityBudgetExceeded) as exc_info:
            enforce_quality_budget(results, _cfg())
        assert exc_info.value.trigger == "stage_b_failure_rate"


class TestDisableFlag:
    def test_no_quality_budget_disables_both(self):
        from sema.pipeline.quality_budget import enforce_quality_budget

        results = [_failed(f"f{i}", "L2 stage_b") for i in range(35)]
        cfg = _cfg(
            quality_budget_max_failure_rate=1.0,
            quality_budget_max_non_contributing_rate=1.0,
        )
        enforce_quality_budget(results, cfg)


class TestExceptionFields:
    def test_stage_b_trigger_carries_breakdown(self):
        from sema.pipeline.quality_budget import (
            QualityBudgetExceeded, enforce_quality_budget,
        )

        results = (
            [_success(f"s{i}") for i in range(18)]
            + [_failed(f"f{i}", "L2 stage_b") for i in range(12)]
        )
        with pytest.raises(QualityBudgetExceeded) as exc_info:
            enforce_quality_budget(results, _cfg())
        err = exc_info.value
        assert err.trigger == "stage_b_failure_rate"
        assert err.failure_rate == pytest.approx(12/30, abs=0.001)
        assert err.threshold == 0.30
        assert err.denominator == 30
        b = err.breakdown
        assert b["stage_b_failed"] == 12
        assert b["succeeded"] == 18
