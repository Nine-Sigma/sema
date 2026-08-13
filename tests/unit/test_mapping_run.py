"""US-012 / G-06: the report must name its subject and describe itself.

``GRAIN_KEY`` is a 5-tuple but ``report_from_store()`` defaulted every filter to
``None``, so the first run under a second policy or vocabulary release would
silently blend two resolvers into one matrix. An evaluation subject is now
required, and a run is written as an immutable self-describing artifact — the
eval-side answer to a store whose grain excludes ``run_id`` and upserts in place.

The verdict is scope-qualified: a bare ``accepted`` over a frequency-selected
head would overclaim about the distinct-code tail it never looked at.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sema.eval.goldset_snapshot_utils import GoldSetHeader
from sema.eval.goldset_source import SourceKind, SourceSpec
from sema.eval.mapping_goldset import GoldSet
from sema.eval.mapping_goldset_utils import (
    Decision,
    GoldLabel,
    GoldRow,
    ResolutionStatus,
    TierState,
)
from sema.eval.mapping_report import build_mapping_report
from sema.eval.mapping_report_utils import AcceptanceVerdict
from sema.eval.mapping_run import (
    EvalRunExistsError,
    EvaluationSubject,
    write_eval_run,
)
from sema.models.planner.lifecycle import Status

pytestmark = pytest.mark.unit


_SUBJECT = EvaluationSubject(
    source_vocabulary="OncoTree",
    target_property_ref="target.condition_occurrence_staging.condition_concept_id",
    resolver_policy_ref="omop.oncotree_condition",
    vocab_release="omop-vocab-2024",
)

_HEADER = GoldSetHeader(
    snapshot_version="2026-08-11-staging",
    snapshot_date="2026-08-11",
    source_of_truth=SourceSpec(
        kind=SourceKind.STAGING,
        table="sema_staging.condition_staging",
        code_column="source_oncotree_code",
        scope_column="source_schema",
        scope_values=("cbioportal_msk_chord_2024",),
    ),
    target_vocabulary="SNOMED",
    target_domain="Condition",
    vocab_release="omop-vocab-2024",
    oracle_source="human curation",
    oracle_version="unlabelled",
    tier_codes=(),
    challenge_codes=(),
    tier_target_row_share=0.95,
    tier_achieved_row_share=0.95,
    min_frozen_tier_row_share=0.90,
    max_unseen_code_share=0.10,
    rows_sha256="abc123",
    universe_sha256="def456",
)


def _row(
    code: str,
    state: TierState,
    label: GoldLabel = GoldLabel.UNLABELLED,
    concept: int | None = None,
    second_reviewer: str | None = None,
) -> GoldRow:
    labelled = label is not GoldLabel.UNLABELLED
    return GoldRow(
        oncotree_code=code,
        gold_concept_id=concept,
        gold_label=label,
        row_count=10,
        tier_state=state,
        target_concept_code=None if concept is None else str(concept),
        curator="dean" if labelled else None,
        review_date="2026-08-11" if labelled else None,
        evidence="OncoTree browser" if labelled else None,
        second_reviewer=second_reviewer,
    )


def _decision(code: str, concept: int) -> Decision:
    return Decision(
        source_code=code,
        concept_id=concept,
        status=Status.auto_accepted,
        resolution_status=ResolutionStatus.RESOLVED,
    )


def _head(n: int, second_reviewer: str | None = None) -> list[GoldRow]:
    return [
        _row(f"H{i}", TierState.IN_TIER, GoldLabel.RESOLVED, 1000 + i, second_reviewer)
        for i in range(n)
    ]


def _decisions(n: int) -> list[Decision]:
    return [_decision(f"H{i}", 1000 + i) for i in range(n)]


# --- evaluation subject -----------------------------------------------------


def test_subject_names_every_fixed_grain_field() -> None:
    assert set(_SUBJECT.as_dict()) == {
        "source_vocabulary",
        "target_property_ref",
        "resolver_policy_ref",
        "vocab_release",
        "unit_of_evaluation",
    }
    assert _SUBJECT.as_dict()["unit_of_evaluation"] == "normalized_source_value"


def test_subject_is_required_to_build_a_store_report() -> None:
    from sema.eval.mapping_report import report_from_store

    with pytest.raises(TypeError):
        report_from_store(object(), Path("x"))  # type: ignore[call-arg,arg-type]


# --- scope-qualified verdict ------------------------------------------------


def test_acceptance_is_never_reported_as_bare_accepted() -> None:
    report = build_mapping_report(GoldSet(_head(12, "sam")), _decisions(12))

    assert report.verdict is AcceptanceVerdict.ACCEPTED
    assert report.verdict.value == "accepted_for_frozen_frequency_head"
    assert "accepted" != report.as_dict()["verdict"]


def test_the_verdict_carries_the_snapshot_version() -> None:
    report = build_mapping_report(
        GoldSet(_head(12, "sam")), _decisions(12), snapshot_version="2026-08-11-staging"
    )

    assert report.as_dict()["snapshot_version"] == "2026-08-11-staging"
    assert "2026-08-11-staging" in report.human_summary()


def test_an_unreviewed_oracle_qualifies_the_verdict_as_unadjudicated() -> None:
    report = build_mapping_report(GoldSet(_head(12)), _decisions(12))

    assert report.verdict is AcceptanceVerdict.ACCEPTED
    assert "unadjudicated" in report.qualifiers
    assert "unadjudicated" in report.human_summary()


def test_a_second_reviewed_oracle_is_not_qualified_unadjudicated() -> None:
    report = build_mapping_report(GoldSet(_head(12, "sam")), _decisions(12))

    assert "unadjudicated" not in report.qualifiers


def test_every_labelled_challenge_code_must_be_second_reviewed() -> None:
    """A wrong gold NO_MAP scores fp_map against a correctly-mapped code."""
    gold = GoldSet(
        _head(12, "sam") + [_row("UESL", TierState.CHALLENGE, GoldLabel.NO_MAP)]
    )

    report = build_mapping_report(gold, _decisions(12))

    assert "unadjudicated" in report.qualifiers


def test_an_entirely_unlabelled_challenge_stratum_stays_unadjudicated() -> None:
    """Filtering to labelled rows first made the challenge check vacuous: with no
    challenge label at all there was nothing to find missing a reviewer."""
    gold = GoldSet(
        _head(12, "sam") + [_row("UESL", TierState.CHALLENGE) for _ in range(1)]
    )

    report = build_mapping_report(gold, _decisions(12), challenge_codes=("UESL",))

    assert "unadjudicated" in report.qualifiers


def test_a_challenge_code_that_sits_in_the_head_still_needs_a_reviewer() -> None:
    """Live case IMMC: declared in the header's challenge list, but derive_states
    resolves it to IN_TIER, so a tier_state check can never demand its review."""
    gold = GoldSet(
        _head(12, "sam")
        + [_row("IMMC", TierState.IN_TIER, GoldLabel.RESOLVED, 777, None)]
    )

    report = build_mapping_report(
        gold, [*_decisions(12), _decision("IMMC", 777)], challenge_codes=("IMMC",)
    )

    assert "unadjudicated" in report.qualifiers


def test_the_head_review_floor_is_the_frozen_head_not_what_has_been_labelled() -> None:
    """3 labelled-and-reviewed head codes cleared a floor of 10 because the floor
    was min(10, len(LABELLED head)) rather than min(10, len(head))."""
    gold = GoldSet(
        _head(3, "sam")
        + [_row(f"U{i}", TierState.IN_TIER) for i in range(134)]
    )

    report = build_mapping_report(gold, _decisions(3))

    assert "unadjudicated" in report.qualifiers


def test_a_small_head_may_be_fully_reviewed_below_the_nominal_floor() -> None:
    """The floor is a sample size, not an absolute: a 4-code head needs 4."""
    report = build_mapping_report(GoldSet(_head(4, "sam")), _decisions(4))

    assert "unadjudicated" not in report.qualifiers


def test_an_out_of_tier_label_alone_does_not_count_as_an_adjudicated_oracle() -> None:
    """`labelled` spanned every row, so one out-of-tier label with zero head
    labels satisfied the non-empty check and both strata checks vacuously."""
    gold = GoldSet(
        [_row("RARE", TierState.OUT_OF_TIER, GoldLabel.RESOLVED, 5, "sam")]
        + [_row(f"U{i}", TierState.IN_TIER) for i in range(137)]
    )

    report = build_mapping_report(gold, [])

    assert "unadjudicated" in report.qualifiers


# --- three strata, reported separately --------------------------------------


def test_the_three_strata_are_reported_separately() -> None:
    gold = GoldSet(
        _head(12, "sam")
        + [_row("UESL", TierState.CHALLENGE, GoldLabel.NO_MAP, second_reviewer="sam")]
        + [_row(f"T{i}", TierState.OUT_OF_TIER) for i in range(30)]
    )

    graded = [
        *_decisions(12),
        Decision(
            source_code="UESL",
            concept_id=None,
            status=Status.auto_accepted,
            resolution_status=ResolutionStatus.NO_MAP,
            no_map_reason="no acceptable target",
        ),
    ]

    payload = build_mapping_report(gold, graded).as_dict()

    assert payload["metrics"]["distinct_code"]["scored"] == 12
    assert payload["metrics"]["challenge"]["scored_codes"] == 1
    assert payload["strata"]["random_tail_sample"]["gating"] is False
    assert 0 < len(payload["strata"]["random_tail_sample"]["codes"]) <= 20


def test_the_random_tail_sample_is_deterministic_per_snapshot() -> None:
    gold = GoldSet(_head(2, "sam") + [_row(f"T{i}", TierState.OUT_OF_TIER) for i in range(30)])

    first = build_mapping_report(gold, _decisions(2), snapshot_version="v1")
    again = build_mapping_report(gold, _decisions(2), snapshot_version="v1")
    other = build_mapping_report(gold, _decisions(2), snapshot_version="v2")

    assert first.tail_sample_codes == again.tail_sample_codes
    assert first.tail_sample_codes != other.tail_sample_codes


def test_a_labelled_tail_code_is_scored_into_its_own_matrix() -> None:
    """The stratum billed as 'the only thing that speaks to confidently wrong in the
    tail' could not: score() dropped every OUT_OF_TIER row, so the report emitted a
    list of codes and a labelled count and no metric at all."""
    gold = GoldSet(
        _head(12, "sam")
        + [_row("RARE", TierState.OUT_OF_TIER, GoldLabel.RESOLVED, 500)]
    )

    report = build_mapping_report(
        gold, [*_decisions(12), _decision("RARE", 999)]  # confidently wrong
    )

    assert report.score.tail_distinct_code.wrong == 1
    assert report.score.tail_distinct_code.mapped_precision == pytest.approx(0.0)
    assert report.score.tail_scored_codes == 1


def test_a_labelled_tail_code_cannot_move_the_gating_matrix() -> None:
    """The §1.5(f) hazard still holds: out-of-tier rows stay out of the primary
    matrix, they just get one of their own."""
    head_only = build_mapping_report(GoldSet(_head(12, "sam")), _decisions(12))
    with_tail = build_mapping_report(
        GoldSet(_head(12, "sam") + [_row("RARE", TierState.OUT_OF_TIER, GoldLabel.RESOLVED, 5)]),
        [*_decisions(12), _decision("RARE", 999)],
    )

    assert with_tail.score.distinct_code.as_dict() == head_only.score.distinct_code.as_dict()
    assert with_tail.verdict is head_only.verdict
    assert with_tail.ungraded_codes == ()


def test_a_retired_code_is_never_scored_even_when_labelled() -> None:
    """It left the declared scope, so there is nothing live to grade it against."""
    gold = GoldSet(_head(12, "sam") + [_row("GBM", TierState.RETIRED, GoldLabel.RESOLVED, 7)])

    report = build_mapping_report(gold, [*_decisions(12), _decision("GBM", 7)])

    assert report.score.tail_scored_codes == 0
    assert "GBM" in report.score.unscored_out_of_scope


def test_the_tail_sample_is_drawn_from_the_universe_manifest() -> None:
    """Sampling only existing gold rows drew from codes that happened to be in a
    PRIOR scope — a biased slice of the tail, not a sample of it."""
    gold = GoldSet(_head(2, "sam") + [_row("LEGACY", TierState.OUT_OF_TIER)])
    manifest = tuple(f"T{i}" for i in range(300))

    report = build_mapping_report(
        gold, _decisions(2), snapshot_version="v1", tail_universe=manifest
    )

    assert len(report.tail_sample_codes) == 20
    assert set(report.tail_sample_codes) <= set(manifest)


def test_the_tail_sample_says_which_codes_are_not_yet_labellable() -> None:
    """A sampled manifest code with no gold row cannot be labelled, so a zero
    labelled count must not read as 'looked at and found clean'."""
    gold = GoldSet(_head(2, "sam") + [_row("T7", TierState.OUT_OF_TIER)])

    report = build_mapping_report(
        gold, _decisions(2), snapshot_version="v1",
        tail_universe=tuple(f"T{i}" for i in range(300)),
    )
    stratum = report.as_dict()["strata"]["random_tail_sample"]

    assert stratum["labelled"] == 0
    assert stratum["without_a_gold_row"] == len(report.tail_sample_codes) - (
        1 if "T7" in report.tail_sample_codes else 0
    )


def test_the_random_draw_and_the_labelled_census_are_separate_keys() -> None:
    """One key carried both, so the matrix named for the RANDOM draw was populated
    from whatever a curator happened to choose to label."""
    gold = GoldSet(
        _head(2, "sam")
        + [_row("RARE", TierState.OUT_OF_TIER, GoldLabel.RESOLVED, 500)]
    )

    payload = build_mapping_report(
        gold,
        [*_decisions(2), _decision("RARE", 999)],
        snapshot_version="v1",
        tail_universe=tuple(f"T{i}" for i in range(300)),
    ).as_dict()

    assert "random_tail" not in payload["strata"]
    draw = payload["strata"]["random_tail_sample"]
    census = payload["strata"]["labelled_tail_census"]
    assert "RARE" not in draw["codes"], "RARE is not in the manifest that was drawn from"
    assert draw["labelled"] == 0
    assert census["scored_codes"] == 1
    assert census["distinct_code"]["wrong"] == 1


def test_the_summary_reports_the_challenge_metrics() -> None:
    """Challenge is the only stratum that can grade NO_MAP; the head's is n/a."""
    gold = GoldSet(
        _head(12, "sam")
        + [_row("UESL", TierState.CHALLENGE, GoldLabel.NO_MAP, second_reviewer="sam")]
    )
    graded = [
        *_decisions(12),
        Decision(
            source_code="UESL",
            concept_id=None,
            status=Status.auto_accepted,
            resolution_status=ResolutionStatus.NO_MAP,
            no_map_reason="no acceptable target",
        ),
    ]

    summary = build_mapping_report(gold, graded, challenge_codes=("UESL",)).human_summary()

    assert "challenge mapped_precision" in summary
    assert "challenge no_map_accuracy   = 100.0%" in summary


def test_the_summary_states_the_declared_challenge_population_honestly() -> None:
    """Live case IMMC: declared challenge AND inside the head, so it is graded in
    the head — 12 declared codes can yield at most 11 challenge rows."""
    gold = GoldSet(
        _head(12, "sam")
        + [_row("IMMC", TierState.IN_TIER, GoldLabel.RESOLVED, 777, "sam")]
        + [_row("GONE", TierState.RETIRED, GoldLabel.NO_MAP, second_reviewer="sam")]
    )

    report = build_mapping_report(
        gold,
        [*_decisions(12), _decision("IMMC", 777)],
        challenge_codes=("IMMC", "GONE", "UESL"),
    )
    summary = report.human_summary()

    assert report.retired_challenge_codes == ("GONE",)
    assert "3 declared" in summary
    assert "IMMC" in summary and "GONE" in summary
    assert report.as_dict()["strata"]["challenge"]["retired"] == ["GONE"]
    assert report.as_dict()["strata"]["challenge"]["also_in_head"] == ["IMMC"]


def test_the_random_tail_never_touches_the_primary_matrix() -> None:
    head_only = build_mapping_report(GoldSet(_head(12, "sam")), _decisions(12))
    with_tail = build_mapping_report(
        GoldSet(_head(12, "sam") + [_row(f"T{i}", TierState.OUT_OF_TIER) for i in range(30)]),
        _decisions(12),
    )

    assert with_tail.score.distinct_code.as_dict() == head_only.score.distinct_code.as_dict()
    assert with_tail.verdict is head_only.verdict


# --- immutable, self-describing run artifact --------------------------------


def test_an_eval_run_records_everything_needed_to_re_verify_it(tmp_path: Path) -> None:
    report = build_mapping_report(
        GoldSet(_head(3, "sam")), _decisions(3), snapshot_version="2026-08-11-staging"
    )

    run_dir = write_eval_run(
        tmp_path,
        run_id="run-1",
        report=report,
        subject=_SUBJECT,
        decisions=_decisions(3),
        header=_HEADER,
        resolver_run_ids=["resolver-run-7"],
    )
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert manifest["run_id"] == "run-1"
    assert manifest["subject"] == _SUBJECT.as_dict()
    assert manifest["gold_set"] == {
        "snapshot_version": "2026-08-11-staging",
        "rows_sha256": "abc123",
        "universe_sha256": "def456",
        "vocab_release": "omop-vocab-2024",
        "target_vocabulary": "SNOMED",
        "target_domain": "Condition",
    }
    assert manifest["resolver_run_ids"] == ["resolver-run-7"], (
        "the store overwrites decisions in place under one grain key, so the run "
        "artifact is the only surviving record of which execution was graded"
    )
    assert manifest["decisions_sha256"]
    assert manifest["decision_count"] == 3
    assert (run_dir / "decisions.jsonl").read_text(encoding="utf-8").count("\n") == 3
    assert json.loads((run_dir / "report.json").read_text(encoding="utf-8"))["verdict"]


def test_an_eval_run_is_never_overwritten(tmp_path: Path) -> None:
    report = build_mapping_report(GoldSet(_head(1, "sam")), _decisions(1))
    args = dict(
        run_id="run-1", report=report, subject=_SUBJECT, decisions=_decisions(1),
        header=_HEADER,
    )
    write_eval_run(tmp_path, **args)  # type: ignore[arg-type]

    with pytest.raises(EvalRunExistsError, match="run-1"):
        write_eval_run(tmp_path, **args)  # type: ignore[arg-type]


def test_a_refused_rewrite_leaves_the_run_directory_intact(tmp_path: Path) -> None:
    """The bare mkdir raised only AFTER a partial write on some paths, leaving a
    half-written run that blocked the retry it forced."""
    report = build_mapping_report(GoldSet(_head(1, "sam")), _decisions(1))
    args = dict(
        run_id="run-1", report=report, subject=_SUBJECT, decisions=_decisions(1),
        header=_HEADER,
    )
    directory = write_eval_run(tmp_path, **args)  # type: ignore[arg-type]
    before = (directory / "run.json").read_text(encoding="utf-8")

    with pytest.raises(EvalRunExistsError):
        write_eval_run(tmp_path, **args)  # type: ignore[arg-type]

    assert (directory / "run.json").read_text(encoding="utf-8") == before
    assert not [p for p in tmp_path.iterdir() if p.name != "run-1"], (
        "a failed write must leave no temporary sibling behind"
    )


def test_a_run_becomes_visible_only_once_it_is_complete(tmp_path: Path) -> None:
    """The directory appears under its run_id atomically, fully written."""
    report = build_mapping_report(GoldSet(_head(1, "sam")), _decisions(1))
    directory = write_eval_run(
        tmp_path, run_id="run-2", report=report, subject=_SUBJECT,
        decisions=_decisions(1), header=_HEADER,
    )

    assert directory == tmp_path / "run-2"
    assert sorted(p.name for p in directory.iterdir()) == [
        "decisions.jsonl", "report.json", "run.json",
    ]
    assert [p.name for p in tmp_path.iterdir()] == ["run-2"]


def test_the_decision_digest_is_order_independent(tmp_path: Path) -> None:
    report = build_mapping_report(GoldSet(_head(3, "sam")), _decisions(3))
    common = dict(report=report, subject=_SUBJECT, header=_HEADER)

    forward = write_eval_run(tmp_path, run_id="a", decisions=_decisions(3), **common)  # type: ignore[arg-type]
    reverse = write_eval_run(
        tmp_path, run_id="b", decisions=list(reversed(_decisions(3))), **common  # type: ignore[arg-type]
    )

    assert json.loads((forward / "run.json").read_text())["decisions_sha256"] == json.loads(
        (reverse / "run.json").read_text()
    )["decisions_sha256"]
