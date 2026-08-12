"""US-002 / G-02+G-02a: two frozen populations, two eligibility predicates.

``acceptance_eligible`` (the frozen frequency head) alone feeds
``coverage_fraction()``, the report totals, and the acceptance gate.
``score_eligible`` (head + challenge stratum) is what gets scored — but the
challenge stratum is reported in its OWN confusion matrix and never merged into
the primary one, because those codes were selected BY the system under test:
folding them into the acceptance population would make benchmark membership move
whenever the resolver changes.

Filtering ``score()`` alone is not enough. ``evaluate_acceptance`` gates on
``coverage_fraction >= 1.0`` and the fraction counted EVERY artifact row, so
out-of-tier and retired rows preserved as ``UNLABELLED`` capped coverage
permanently below 1.0 — fully labelling the head still read "provisional".
"""

from __future__ import annotations

import pytest

from sema.eval.mapping_goldset import GoldSet, score
from sema.eval.mapping_goldset_utils import (
    Decision,
    GoldLabel,
    GoldRow,
    ResolutionStatus,
    TierState,
)
from sema.eval.mapping_report import build_mapping_report
from sema.eval.mapping_report_utils import AcceptanceVerdict
from sema.models.planner.lifecycle import Status

pytestmark = pytest.mark.unit


def _row(
    code: str,
    state: TierState,
    label: GoldLabel = GoldLabel.UNLABELLED,
    concept: int | None = None,
    row_count: int = 10,
) -> GoldRow:
    return GoldRow(
        oncotree_code=code,
        gold_concept_id=concept,
        gold_label=label,
        row_count=row_count,
        tier_state=state,
        target_concept_code=None if concept is None else str(concept),
        curator=None if label is GoldLabel.UNLABELLED else "dean",
        review_date=None if label is GoldLabel.UNLABELLED else "2026-08-11",
        evidence=None if label is GoldLabel.UNLABELLED else "OncoTree browser",
    )


def _resolved(code: str, concept: int) -> Decision:
    return Decision(
        source_code=code,
        concept_id=concept,
        status=Status.auto_accepted,
        resolution_status=ResolutionStatus.RESOLVED,
    )


def _no_map(code: str) -> Decision:
    return Decision(
        source_code=code,
        concept_id=None,
        status=Status.auto_accepted,
        resolution_status=ResolutionStatus.NO_MAP,
        no_map_reason="no acceptable target",
    )


def _head_labelled() -> list[GoldRow]:
    return [
        _row("LUAD", TierState.IN_TIER, GoldLabel.RESOLVED, 45768916, row_count=1000),
        _row("COAD", TierState.IN_TIER, GoldLabel.RESOLVED, 4180790, row_count=500),
    ]


def _tail_unlabelled() -> list[GoldRow]:
    return [
        _row("RARE", TierState.OUT_OF_TIER, row_count=2),
        _row("GBM", TierState.RETIRED, row_count=7),
    ]


# --- coverage denominator ---------------------------------------------------


def test_coverage_counts_only_the_acceptance_population() -> None:
    gold = GoldSet(_head_labelled() + _tail_unlabelled())

    assert gold.coverage_fraction() == pytest.approx(1.0)
    assert gold.labelled_count == 2
    assert gold.total_eligible_codes == 2


def test_an_in_tier_gap_still_drags_coverage_down() -> None:
    gold = GoldSet([*_head_labelled(), _row("LUSC", TierState.IN_TIER)] + _tail_unlabelled())

    assert gold.coverage_fraction() == pytest.approx(2 / 3)


def test_unlabelled_codes_are_in_tier_work_not_accounted_for_rows() -> None:
    gold = GoldSet([*_head_labelled(), _row("LUSC", TierState.IN_TIER)] + _tail_unlabelled())

    assert gold.unlabelled_codes() == ["LUSC"]
    assert gold.out_of_tier_codes() == ["RARE"]
    assert gold.retired_codes() == ["GBM"]


def test_a_fully_labelled_head_reaches_accepted_despite_an_unlabelled_tail() -> None:
    """The single change that decides whether the eval can ever go green."""
    gold = GoldSet(_head_labelled() + _tail_unlabelled())
    decisions = [_resolved("LUAD", 45768916), _resolved("COAD", 4180790)]

    report = build_mapping_report(gold, decisions)

    assert report.verdict is AcceptanceVerdict.ACCEPTED
    assert report.total_codes == 2, "report totals must share the gate's denominator"
    assert report.labelled_count == 2


def test_challenge_rows_do_not_block_acceptance() -> None:
    gold = GoldSet([*_head_labelled(), _row("UESL", TierState.CHALLENGE)] + _tail_unlabelled())

    report = build_mapping_report(gold, [_resolved("LUAD", 45768916), _resolved("COAD", 4180790)])

    assert report.verdict is AcceptanceVerdict.ACCEPTED


# --- scoring eligibility ----------------------------------------------------


def test_out_of_tier_and_retired_codes_are_never_scored() -> None:
    """The OUT_OF_TIER-as-a-GoldLabel hazard: they must not become fp_map."""
    gold = [
        *_head_labelled(),
        _row("RARE", TierState.OUT_OF_TIER, GoldLabel.NO_MAP, row_count=2),
        _row("GBM", TierState.RETIRED, GoldLabel.NO_MAP, row_count=7),
    ]
    decisions = [
        _resolved("LUAD", 45768916),
        _resolved("COAD", 4180790),
        _resolved("RARE", 999),
        _resolved("GBM", 888),
    ]

    report = score(gold, decisions)

    assert report.distinct_code.fp_map == 0
    assert report.distinct_code.mapped_precision == pytest.approx(1.0)
    assert report.scored_codes == 2
    assert report.unscored_out_of_scope == ["GBM", "RARE"]


def test_a_challenge_row_cannot_move_any_primary_matrix_value() -> None:
    """G-02a's test: the two populations are frozen and never merged."""
    head_only = score(_head_labelled(), [_resolved("LUAD", 45768916), _resolved("COAD", 4180790)])

    with_challenge = score(
        [*_head_labelled(), _row("UESL", TierState.CHALLENGE, GoldLabel.NO_MAP, row_count=3)],
        [_resolved("LUAD", 45768916), _resolved("COAD", 4180790), _resolved("UESL", 42)],
    )

    assert with_challenge.distinct_code.as_dict() == head_only.distinct_code.as_dict()
    assert with_challenge.row_weighted.as_dict() == head_only.row_weighted.as_dict()
    assert with_challenge.per_bucket.keys() == head_only.per_bucket.keys()


def test_the_challenge_stratum_gets_its_own_matrix() -> None:
    gold = [
        *_head_labelled(),
        _row("UESL", TierState.CHALLENGE, GoldLabel.NO_MAP, row_count=3),
        _row("BTOV", TierState.CHALLENGE, GoldLabel.NO_MAP, row_count=1),
    ]
    decisions = [
        _resolved("LUAD", 45768916),
        _resolved("COAD", 4180790),
        _no_map("UESL"),
        _resolved("BTOV", 42),
    ]

    report = score(gold, decisions)

    assert report.challenge_scored_codes == 2
    assert report.challenge_distinct_code.tn == 1
    assert report.challenge_distinct_code.fp_map == 1
    assert report.challenge_row_weighted.tn == 3


def test_no_map_accuracy_is_computable_only_via_the_challenge_stratum() -> None:
    """All 7 live resolver NO_MAP codes rank outside the head; head-only this is None."""
    gold = [*_head_labelled(), _row("UESL", TierState.CHALLENGE, GoldLabel.NO_MAP, row_count=3)]
    decisions = [_resolved("LUAD", 45768916), _resolved("COAD", 4180790), _no_map("UESL")]

    report = score(gold, decisions)

    assert report.distinct_code.no_map_accuracy is None
    assert report.challenge_distinct_code.no_map_accuracy == pytest.approx(1.0)


def test_out_of_tier_is_not_a_gold_label() -> None:
    """classify_cell reads every non-RESOLVED label as gold-NO_MAP; tier is a row attribute."""
    assert {member.value for member in GoldLabel} == {"RESOLVED", "NO_MAP", "UNLABELLED"}


# --- one decision per eligible code -----------------------------------------


def test_duplicate_decisions_for_an_eligible_code_raise() -> None:
    """Last-wins would silently blend two resolvers, ordered by DuckDB row order."""
    decisions = [_resolved("LUAD", 45768916), _resolved("LUAD", 1)]

    with pytest.raises(ValueError, match="LUAD"):
        score(_head_labelled(), decisions)


def test_gate_d_lite_does_not_count_a_retired_code_as_a_coverage_miss() -> None:
    """The fifth call site: a code that left the scope is not staging debt."""
    from sema.eval.staging_qa_utils import StagingRow, check_no_map_accounting

    gold = GoldSet([*_head_labelled(), _row("GBM", TierState.RETIRED, row_count=7)])
    staged = [
        StagingRow(source_value="LUAD", target_value=45768916, resolution_status="RESOLVED"),
        StagingRow(source_value="COAD", target_value=4180790, resolution_status="RESOLVED"),
    ]

    check = check_no_map_accounting(staged, gold)

    assert check.passed
    assert "GBM" not in check.details["staged_no_map"]
    assert gold.unlabelled_codes() == []


def test_duplicate_decisions_outside_the_scored_populations_are_ignored() -> None:
    gold = [*_head_labelled(), _row("RARE", TierState.OUT_OF_TIER, row_count=2)]
    decisions = [
        _resolved("LUAD", 45768916),
        _resolved("COAD", 4180790),
        _resolved("RARE", 1),
        _resolved("RARE", 2),
    ]

    assert score(gold, decisions).scored_codes == 2
