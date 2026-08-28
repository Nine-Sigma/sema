"""US-002 / G-03: row_count drift is REPORTED with a number, never asserted to zero.

``row_count`` is a scoring weight, not truth. Asserting it equal to live data
guaranteed a red suite after every ingest; the answer is a drift figure plus a
separately-named freshness check, not a tighter assertion.

Two traps the formula must avoid: signed deltas let a doubled code cancel a
halved one and report ~0, and an aggregate taken over codes with no frozen
``old`` count silently drops them from the denominator.
"""

from __future__ import annotations

import pytest

from sema.eval.goldset_drift import goldset_drift_report
from sema.eval.goldset_snapshot import GoldSetSnapshot
from sema.eval.goldset_snapshot_utils import GoldSetHeader, UniverseEntry
from sema.eval.goldset_source import SourceKind, SourceSpec
from sema.eval.mapping_goldset_utils import GoldLabel, GoldRow, TierState

pytestmark = pytest.mark.unit


_SPEC = SourceSpec(
    kind=SourceKind.STAGING,
    table="sema_staging.condition_staging",
    code_column="source_oncotree_code",
    scope_column="source_schema",
    scope_values=("study_a",),
)


def _snapshot(**overrides: object) -> GoldSetSnapshot:
    universe = (
        UniverseEntry("LUAD", 100),
        UniverseEntry("COAD", 100),
        UniverseEntry("RARE", 10),
    )
    header = GoldSetHeader(
        snapshot_version="v1",
        snapshot_date="2026-08-11",
        source_of_truth=_SPEC,
        target_vocabulary="SNOMED",
        target_domain="Condition",
        vocab_release="omop-vocab-2024",
        oracle_source="human curation",
        oracle_version="unlabelled",
        tier_codes=("LUAD", "COAD"),
        challenge_codes=(),
        tier_target_row_share=0.95,
        tier_achieved_row_share=200 / 210,
        min_frozen_tier_row_share=0.90,
        max_unseen_code_share=0.10,
        **overrides,  # type: ignore[arg-type]
    )
    rows = [
        GoldRow("LUAD", None, GoldLabel.UNLABELLED, 100, tier_state=TierState.IN_TIER),
        GoldRow("COAD", None, GoldLabel.UNLABELLED, 100, tier_state=TierState.IN_TIER),
        GoldRow("RARE", None, GoldLabel.UNLABELLED, 10, tier_state=TierState.OUT_OF_TIER),
    ]
    return GoldSetSnapshot(header=header, rows=rows, universe=universe)


def test_opposing_deltas_do_not_cancel() -> None:
    """A doubled LUAD offsetting a halved COAD must not report ~0 drift."""
    report = goldset_drift_report(_snapshot(), {"LUAD": 200, "COAD": 0, "RARE": 10}, _SPEC)

    assert report.universe_drift == pytest.approx((100 + 100 + 0) / 210)


def test_no_drift_reports_zero() -> None:
    report = goldset_drift_report(_snapshot(), {"LUAD": 100, "COAD": 100, "RARE": 10}, _SPEC)

    assert report.universe_drift == pytest.approx(0.0)
    assert report.eligible_drift == pytest.approx(0.0)
    assert not report.codes_added and not report.codes_disappeared


def test_eligible_and_universe_drift_are_reported_separately() -> None:
    """An aggregate over codes that are never labelled or scored decides nothing."""
    report = goldset_drift_report(_snapshot(), {"LUAD": 100, "COAD": 100, "RARE": 1000}, _SPEC)

    assert report.eligible_drift == pytest.approx(0.0)
    assert report.universe_drift == pytest.approx(990 / 210)


def test_added_and_disappeared_codes_are_listed_not_averaged() -> None:
    report = goldset_drift_report(_snapshot(), {"LUAD": 100, "COAD": 100, "NEW": 5}, _SPEC)

    assert report.codes_added == ["NEW"]
    assert report.codes_disappeared == ["RARE"]
    assert report.universe_drift == pytest.approx(0.0), "only codes in BOTH snapshots"


def test_a_scope_change_is_flagged_not_averaged() -> None:
    """Per-code drift across different scopes is meaningless."""
    other = SourceSpec(
        kind=SourceKind.STAGING,
        table="sema_staging.condition_staging",
        code_column="source_oncotree_code",
        scope_column="source_schema",
        scope_values=("study_a", "study_b"),
    )

    report = goldset_drift_report(_snapshot(), {"LUAD": 100, "COAD": 100, "RARE": 10}, other)

    assert report.scope_changed
    assert report.universe_drift is None
    assert report.eligible_drift is None


def test_per_code_deltas_are_signed_and_addressable() -> None:
    report = goldset_drift_report(_snapshot(), {"LUAD": 120, "COAD": 90, "RARE": 10}, _SPEC)

    assert report.per_code["LUAD"] == 20
    assert report.per_code["COAD"] == -10
    assert "RARE" not in report.per_code


# --- benchmark freshness (may red; snapshot integrity may not) --------------


def test_a_representative_benchmark_is_fresh() -> None:
    report = goldset_drift_report(_snapshot(), {"LUAD": 100, "COAD": 100, "RARE": 10}, _SPEC)

    assert report.frozen_tier_row_share == pytest.approx(200 / 210)
    assert report.unseen_code_share == pytest.approx(0.0)
    assert not report.is_stale


def test_an_ingest_that_swamps_the_frozen_tier_reports_stale() -> None:
    """A large enough ingest MUST trip this — that is why it is a separate contract."""
    report = goldset_drift_report(
        _snapshot(), {"LUAD": 100, "COAD": 100, "RARE": 10, "FLOOD": 5000}, _SPEC
    )

    assert report.frozen_tier_row_share < 0.90
    assert report.is_stale
    assert "re-tier" in report.staleness_reason


def test_too_many_unseen_codes_reports_stale() -> None:
    observed = {"LUAD": 100, "COAD": 100, "RARE": 10}
    observed.update({f"NEW{i}": 1 for i in range(20)})

    report = goldset_drift_report(_snapshot(), observed, _SPEC)

    assert report.unseen_code_share > 0.10
    assert report.is_stale


def test_the_report_is_json_serializable() -> None:
    report = goldset_drift_report(_snapshot(), {"LUAD": 120, "COAD": 90}, _SPEC)

    payload = report.as_dict()
    assert payload["scope_changed"] is False
    assert payload["freshness"]["is_stale"] in (True, False)
    assert payload["codes_disappeared"] == ["RARE"]
