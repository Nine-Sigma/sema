"""aggregate_report distinguishes succeeded full vs partial; per-tier counts."""
from __future__ import annotations

import pytest

from sema.pipeline.build import TableResult, aggregate_report

pytestmark = pytest.mark.unit


class TestAggregateReportFullPartial:
    def test_distinguishes_full_and_partial(self):
        results = [
            TableResult.success(
                "t1", entities=1, properties=1,
                partial=False, metadata_tier="rich",
            ),
            TableResult.success(
                "t2", entities=1, properties=1,
                partial=True, metadata_tier="name_only",
            ),
            TableResult.success(
                "t3", entities=1, properties=1,
                partial=True, metadata_tier="sparse",
            ),
        ]
        report = aggregate_report(results)
        assert report["tables_processed"] == 3
        assert report["tables_succeeded_full"] == 1
        assert report["tables_succeeded_partial"] == 2

    def test_metadata_tier_counts(self):
        results = [
            TableResult.success("t1", metadata_tier="rich"),
            TableResult.success("t2", metadata_tier="rich"),
            TableResult.success("t3", metadata_tier="sparse"),
            TableResult.failed("t4", "L2 stage_b", "fail",
                               metadata_tier="name_only"),
        ]
        report = aggregate_report(results)
        assert report["metadata_tier_counts"] == {
            "rich": 2, "sparse": 1, "name_only": 1,
        }


class TestPartialCountsAsSucceededForBudget:
    """B_PARTIAL outcomes count as 'succeeded' for graph-health budget.
    The aggregated report's `tables_processed` includes partials so the
    Stage B budget denominator/numerator semantics are consistent."""

    def test_partial_counts_in_processed(self):
        results = [
            TableResult.success("t1", partial=True, metadata_tier="sparse"),
        ]
        report = aggregate_report(results)
        assert report["tables_processed"] == 1
        assert report["tables_failed"] == 0
