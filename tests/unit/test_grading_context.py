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
from sema.resolve.value_mapping_store_utils import ValueMapping

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


def _snapshot(
    rows: list[GoldRow] | None = None, header: GoldSetHeader | None = None
) -> GoldSetSnapshot:
    return GoldSetSnapshot(
        header=header or _header(),
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


def _mapping(code: str, release: str) -> ValueMapping:
    return ValueMapping(
        source_vocabulary="OncoTree",
        normalized_source_value=code,
        target_property_ref="omop.condition_occurrence.condition_concept_id",
        target_field="condition_concept_id",
        vocab_binding="SNOMED/Condition",
        concept_id=45768916,
        vocab_release=release,
        valid_start=None,
        valid_end=None,
        resolution_status=ResolutionStatus.RESOLVED,
        no_map_reason=None,
        confidence=1.0,
        status=Status.auto_accepted,
        resolver_policy_ref="policy-v1",
        run_id="run-1",
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


# --- the pin needs something to pin to --------------------------------------


def test_a_snapshot_with_no_declared_release_cannot_grade_its_own_labels() -> None:
    """An undeclared release silently skipped the pin altogether.

    ``if context.vocab_release and ...`` reads as "check when we can", which made
    a header missing the one field the pin is built on indistinguishable from a
    header that matched. Labels without a release are ungradable, not universally
    gradable.
    """
    context = GradingContext.from_snapshot(_snapshot(header=_header(vocab_release="")))

    with pytest.raises(GradingReleaseError, match="declares no vocab_release"):
        report_for_snapshot(context, [], graded_release="omop-vocab-2024")


def test_an_empty_context_still_grades_without_a_release() -> None:
    """No oracle, no pin — the one case where an absent release is honest."""
    report = report_for_snapshot(
        GradingContext.empty(), [_decision("LUAD", 1)], graded_release="anything"
    )

    assert report.labelled_count == 0


# --- the graded release is read off the rows, not off a flag ----------------


def test_graded_release_of_reads_the_release_the_rows_carry() -> None:
    from sema.eval.mapping_report import graded_release_of

    rows = [_mapping("LUAD", "omop-vocab-2024"), _mapping("COAD", "omop-vocab-2024")]

    assert graded_release_of(rows, fallback="ignored") == "omop-vocab-2024"


def test_graded_release_of_falls_back_only_when_nothing_was_graded() -> None:
    from sema.eval.mapping_report import graded_release_of

    assert graded_release_of([], fallback="omop-vocab-2024") == "omop-vocab-2024"


def test_two_releases_in_one_decision_set_are_refused_not_sampled() -> None:
    """Reading ``run_mappings[0]`` pinned the first row and graded all of them.

    That is the blend ``EvaluationSubject`` exists to prevent, arriving through
    the pin that was supposed to catch it.
    """
    from sema.eval.mapping_report import graded_release_of

    rows = [_mapping("LUAD", "omop-vocab-2024"), _mapping("COAD", "omop-vocab-2025")]

    with pytest.raises(GradingReleaseError, match="omop-vocab-2025"):
        graded_release_of(rows, fallback="omop-vocab-2024")
