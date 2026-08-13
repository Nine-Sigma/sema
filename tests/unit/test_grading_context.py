"""US-012: one hardened grading path — :class:`GradingContext`.

The hole this closes: ``sema eval goldset mapping-report`` loaded through
``load_snapshot`` and passed ``challenge_codes`` + ``tail_universe``, while
``sema fit --strict`` parsed bare JSONL and passed neither. Two paths graded the
same resolver against the same oracle and disagreed about which strata existed.
``GradingContext`` is the single carrier of the snapshot's declaration and
``report_for_snapshot`` the single entry point that honours it.
"""

from __future__ import annotations

import pytest

from sema.eval.goldset_snapshot import GoldSetSnapshot
from sema.eval.goldset_snapshot_utils import GoldSetHeader, UniverseEntry
from sema.eval.goldset_source import SourceKind, SourceSpec
from sema.eval.mapping_goldset_utils import (
    Decision,
    GoldLabel,
    GoldRow,
    ResolutionStatus,
    TierState,
)
from sema.eval.mapping_report import (
    GradingContext,
    GradingReleaseError,
    report_for_snapshot,
)
from sema.models.planner.lifecycle import Status

pytestmark = pytest.mark.unit


_SPEC = SourceSpec(
    kind=SourceKind.STAGING,
    table="sema_staging.condition_staging",
    code_column="source_oncotree_code",
    scope_column="source_schema",
    scope_values=("cbioportal_study_a",),
)

_UNIVERSE = (
    UniverseEntry("LUAD", 100),
    UniverseEntry("COAD", 50),
    UniverseEntry("ODD", 2),
    UniverseEntry("RARE", 1),
)


def _header(**overrides: object) -> GoldSetHeader:
    base = dict(
        snapshot_version="ctx-v1",
        snapshot_date="2026-08-12",
        source_of_truth=_SPEC,
        target_vocabulary="SNOMED",
        target_domain="Condition",
        vocab_release="omop-vocab-2024",
        oracle_source="human curation",
        oracle_version="unlabelled",
        tier_codes=("LUAD", "COAD"),
        challenge_codes=("ODD",),
        tier_target_row_share=0.95,
        tier_achieved_row_share=150 / 153,
        min_frozen_tier_row_share=0.90,
        max_unseen_code_share=0.10,
    )
    base.update(overrides)
    return GoldSetHeader(**base)  # type: ignore[arg-type]


def _labelled(code: str, concept: int | None, state: TierState, count: int) -> GoldRow:
    return GoldRow(
        oncotree_code=code,
        gold_concept_id=concept,
        gold_label=GoldLabel.RESOLVED if concept else GoldLabel.NO_MAP,
        row_count=count,
        tier_state=state,
        target_concept_code=str(concept) if concept else None,
        curator="alice",
        review_date="2026-08-12",
        evidence="OncoTree browser",
    )


def _snapshot(rows: list[GoldRow] | None = None) -> GoldSetSnapshot:
    return GoldSetSnapshot(
        header=_header(),
        rows=rows
        or [
            GoldRow("LUAD", None, GoldLabel.UNLABELLED, 100, tier_state=TierState.IN_TIER),
            GoldRow("COAD", None, GoldLabel.UNLABELLED, 50, tier_state=TierState.IN_TIER),
            GoldRow("ODD", None, GoldLabel.UNLABELLED, 2, tier_state=TierState.CHALLENGE),
        ],
        universe=_UNIVERSE,
    )


def _decision(
    code: str, concept: int | None, res: ResolutionStatus = ResolutionStatus.RESOLVED
) -> Decision:
    return Decision(
        source_code=code,
        concept_id=concept,
        status=Status.auto_accepted,
        resolution_status=res,
        no_map_reason="dead end" if res is ResolutionStatus.NO_MAP else None,
    )


# --- the context carries the whole declaration ------------------------------


def test_from_snapshot_carries_the_declared_populations() -> None:
    context = GradingContext.from_snapshot(_snapshot())
    assert context.snapshot_version == "ctx-v1"
    assert context.challenge_codes == ("ODD",)
    assert context.vocab_release == "omop-vocab-2024"
    assert [r.oncotree_code for r in context.gold.rows] == ["LUAD", "COAD", "ODD"]


def test_tail_universe_is_the_manifest_minus_both_frozen_populations() -> None:
    context = GradingContext.from_snapshot(_snapshot())
    # RARE is the only manifest code outside the head and the challenge stratum.
    assert context.tail_universe == ("RARE",)


def test_empty_context_grades_without_a_snapshot() -> None:
    context = GradingContext.empty()
    assert context.gold.rows == []
    assert context.challenge_codes == ()
    assert context.tail_universe is None
    assert context.vocab_release == ""


# --- the release pin lives in the library, not in one CLI -------------------


def test_report_for_snapshot_refuses_to_grade_a_foreign_release() -> None:
    context = GradingContext.from_snapshot(_snapshot())
    with pytest.raises(GradingReleaseError) as exc:
        report_for_snapshot(context, [], graded_release="omop-vocab-2099")
    assert "omop-vocab-2024" in str(exc.value)
    assert "omop-vocab-2099" in str(exc.value)


def test_report_for_snapshot_refuses_an_unstated_release() -> None:
    context = GradingContext.from_snapshot(_snapshot())
    with pytest.raises(GradingReleaseError):
        report_for_snapshot(context, [], graded_release="")


def test_empty_context_pins_nothing() -> None:
    report = report_for_snapshot(
        GradingContext.empty(), [_decision("LUAD", 1)], graded_release="anything"
    )
    assert report.labelled_count == 0


# --- the strata the JSONL path used to drop ---------------------------------


def test_challenge_stratum_is_scored_through_the_context() -> None:
    rows = [
        _labelled("LUAD", 45768916, TierState.IN_TIER, 100),
        _labelled("COAD", 4180790, TierState.IN_TIER, 50),
        _labelled("ODD", None, TierState.CHALLENGE, 2),
    ]
    context = GradingContext.from_snapshot(_snapshot(rows))
    report = report_for_snapshot(
        context,
        [
            _decision("LUAD", 45768916),
            _decision("COAD", 4180790),
            _decision("ODD", 12345),
        ],
        graded_release="omop-vocab-2024",
    )
    # gold says NO_MAP, the resolver mapped it: fp_map in the challenge matrix.
    assert report.score.challenge_distinct_code.fp_map == 1
    assert report.has_labelled_contradiction() is True
    assert report.contradiction_strata() == ("challenge",)


def test_tail_sample_is_drawn_from_the_manifest_tail() -> None:
    context = GradingContext.from_snapshot(_snapshot())
    report = report_for_snapshot(context, [], graded_release="omop-vocab-2024")
    assert report.tail_sample_codes == ("RARE",)
