"""Tier-keyed B_PARTIAL admission floor (Section 5b)."""
from __future__ import annotations

import pytest

from sema.engine.stage_utils import determine_b_status
from sema.models.stages import StageBCoverage, UnresolvedColumn

pytestmark = pytest.mark.unit


def _cov(pct: float, classified: int = None, total: int = 10) -> StageBCoverage:
    if classified is None:
        classified = int(round(pct * total))
    return StageBCoverage(classified=classified, total=total, pct=pct)


class TestBSuccessPreservedAcrossTiers:
    def test_rich_b_success_unchanged(self):
        st = determine_b_status(
            raw_coverage=_cov(1.0),
            critical_coverage=_cov(1.0),
            unresolved=[],
            metadata_tier="rich",
        )
        assert st == "B_SUCCESS"

    def test_name_only_b_success_unchanged(self):
        st = determine_b_status(
            raw_coverage=_cov(1.0),
            critical_coverage=_cov(1.0),
            unresolved=[],
            metadata_tier="name_only",
        )
        assert st == "B_SUCCESS"


class TestRichFloorPreserved:
    def test_rich_at_0_85_with_unresolved_partial(self):
        st = determine_b_status(
            raw_coverage=_cov(0.85),
            critical_coverage=_cov(1.0),
            unresolved=[
                UnresolvedColumn(
                    column="x", reason="execution_failure", tier="peripheral",
                ),
                UnresolvedColumn(
                    column="y", reason="execution_failure", tier="peripheral",
                ),
            ],
            metadata_tier="rich",
        )
        assert st == "B_PARTIAL"

    def test_rich_at_0_65_b_failed(self):
        st = determine_b_status(
            raw_coverage=_cov(0.65),
            critical_coverage=_cov(1.0),
            unresolved=[],
            metadata_tier="rich",
        )
        assert st == "B_FAILED"


class TestSparseAndNameOnlyLoweredFloor:
    def test_sparse_at_0_65_partial(self):
        st = determine_b_status(
            raw_coverage=_cov(0.65),
            critical_coverage=_cov(1.0),
            unresolved=[],
            metadata_tier="sparse",
        )
        assert st == "B_PARTIAL"

    def test_name_only_at_0_62_partial(self):
        st = determine_b_status(
            raw_coverage=_cov(0.62),
            critical_coverage=_cov(1.0),
            unresolved=[],
            metadata_tier="name_only",
        )
        assert st == "B_PARTIAL"

    def test_name_only_below_floor_b_failed(self):
        st = determine_b_status(
            raw_coverage=_cov(0.55),
            critical_coverage=_cov(1.0),
            unresolved=[],
            metadata_tier="name_only",
        )
        assert st == "B_FAILED"


class TestCriticalFloorUnchanged:
    def test_name_only_with_crit_below_1_b_failed(self):
        st = determine_b_status(
            raw_coverage=_cov(0.62),
            critical_coverage=_cov(0.90),
            unresolved=[],
            metadata_tier="name_only",
        )
        assert st == "B_FAILED"


class TestConfigurableFloor:
    def test_partial_floor_bumped_reclassifies_sparse(self):
        st = determine_b_status(
            raw_coverage=_cov(0.65),
            critical_coverage=_cov(1.0),
            unresolved=[],
            metadata_tier="sparse",
            partial_coverage_floor=0.70,
        )
        assert st == "B_FAILED"

    def test_partial_floor_bumped_does_not_change_rich(self):
        # Rich still uses its built-in 0.75 floor regardless of param
        st = determine_b_status(
            raw_coverage=_cov(0.85),
            critical_coverage=_cov(1.0),
            unresolved=[
                UnresolvedColumn(
                    column="x", reason="execution_failure", tier="peripheral",
                ),
            ],
            metadata_tier="rich",
            partial_coverage_floor=0.30,
        )
        assert st == "B_PARTIAL"


class TestBackwardsCompatibility:
    """Existing call sites pass no metadata_tier — must still work."""

    def test_omitting_tier_uses_legacy_rich_floor(self):
        st = determine_b_status(
            raw_coverage=_cov(0.85),
            critical_coverage=_cov(1.0),
            unresolved=[
                UnresolvedColumn(
                    column="x", reason="execution_failure", tier="peripheral",
                ),
            ],
        )
        assert st == "B_PARTIAL"
