"""Run-level quality budget with two triggers, one exception.

Stateless check post-`_collect_results`, pre-`run_fk_detection`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal

from sema.models.config import BuildConfig
from sema.pipeline.build import TableResult


_STAGE_B_LABEL = "L2 stage_b"
_STAGE_A_LABEL = "L2 stage_a"
_CIRCUIT_LABEL = "circuit_breaker"
_RESUME_PREFIX = "resume:"


@dataclass
class Breakdown:
    succeeded: int = 0
    stage_b_failed: int = 0
    circuit_open: int = 0
    stage_a_failed: int = 0
    other_failed: int = 0
    skipped_resume: int = 0
    skipped_other: int = 0

    @property
    def all_failed(self) -> int:
        return (
            self.stage_b_failed + self.circuit_open
            + self.stage_a_failed + self.other_failed
        )


class QualityBudgetExceeded(Exception):
    def __init__(
        self,
        *,
        trigger: Literal["stage_b_failure_rate", "run_non_contributing_rate"],
        failure_rate: float,
        threshold: float,
        denominator: int,
        breakdown: Breakdown,
    ):
        self.trigger = trigger
        self.failure_rate = failure_rate
        self.threshold = threshold
        self.denominator = denominator
        self.breakdown = asdict(breakdown)
        super().__init__(
            f"Quality budget exceeded ({trigger}): rate={failure_rate:.3f}, "
            f"threshold={threshold:.3f}, denominator={denominator}, "
            f"breakdown={self.breakdown}"
        )


def compute_breakdown(results: list[TableResult]) -> Breakdown:
    b = Breakdown()
    for r in results:
        if r.status == "success":
            b.succeeded += 1
        elif r.status == "failed":
            stage = r.failed_stage or ""
            if stage == _STAGE_B_LABEL:
                b.stage_b_failed += 1
            elif stage == _STAGE_A_LABEL:
                b.stage_a_failed += 1
            elif stage == _CIRCUIT_LABEL:
                b.circuit_open += 1
            else:
                b.other_failed += 1
        elif r.status == "skipped":
            reason = r.skip_reason or ""
            if reason.startswith(_RESUME_PREFIX):
                b.skipped_resume += 1
            else:
                b.skipped_other += 1
    return b


def _check_stage_b_trigger(
    b: Breakdown, config: BuildConfig,
) -> None:
    graph_contributing = (
        b.succeeded + b.stage_b_failed + b.skipped_resume
    )
    if graph_contributing < config.quality_budget_min_processed:
        return
    rate = b.stage_b_failed / graph_contributing
    if rate > config.quality_budget_max_failure_rate:
        raise QualityBudgetExceeded(
            trigger="stage_b_failure_rate",
            failure_rate=rate,
            threshold=config.quality_budget_max_failure_rate,
            denominator=graph_contributing,
            breakdown=b,
        )


def _check_run_reliability_trigger(
    b: Breakdown, config: BuildConfig,
) -> None:
    attempted = b.succeeded + b.all_failed + b.skipped_other
    if attempted < config.quality_budget_min_processed:
        return
    non_contributing = (
        b.stage_a_failed + b.circuit_open + b.other_failed
    )
    rate = non_contributing / attempted
    if rate > config.quality_budget_max_non_contributing_rate:
        raise QualityBudgetExceeded(
            trigger="run_non_contributing_rate",
            failure_rate=rate,
            threshold=config.quality_budget_max_non_contributing_rate,
            denominator=attempted,
            breakdown=b,
        )


def enforce_quality_budget(
    results: list[TableResult], config: BuildConfig,
) -> None:
    """Raise QualityBudgetExceeded when either trigger fires.

    Stable ordering: Stage B graph-health is checked first.
    """
    b = compute_breakdown(results)
    _check_stage_b_trigger(b, config)
    _check_run_reliability_trigger(b, config)
