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
from sema.eval.mapping_run import EvaluationSubject, write_eval_run
from sema.models.planner.lifecycle import Status

pytestmark = pytest.mark.unit


_SUBJECT = EvaluationSubject(
    source_vocabulary="OncoTree",
    target_property_ref="target.condition_occurrence_staging.condition_concept_id",
    resolver_policy_ref="omop.oncotree_condition",
    vocab_release="omop-vocab-2024",
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
    assert payload["strata"]["random_tail"]["gating"] is False
    assert 0 < len(payload["strata"]["random_tail"]["codes"]) <= 20


def test_the_random_tail_sample_is_deterministic_per_snapshot() -> None:
    gold = GoldSet(_head(2, "sam") + [_row(f"T{i}", TierState.OUT_OF_TIER) for i in range(30)])

    first = build_mapping_report(gold, _decisions(2), snapshot_version="v1")
    again = build_mapping_report(gold, _decisions(2), snapshot_version="v1")
    other = build_mapping_report(gold, _decisions(2), snapshot_version="v2")

    assert first.tail_sample_codes == again.tail_sample_codes
    assert first.tail_sample_codes != other.tail_sample_codes


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
        gold_rows_sha256="abc123",
        universe_sha256="def456",
    )
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert manifest["run_id"] == "run-1"
    assert manifest["subject"] == _SUBJECT.as_dict()
    assert manifest["gold_set"] == {
        "snapshot_version": "2026-08-11-staging",
        "rows_sha256": "abc123",
        "universe_sha256": "def456",
    }
    assert manifest["decisions_sha256"]
    assert manifest["decision_count"] == 3
    assert (run_dir / "decisions.jsonl").read_text(encoding="utf-8").count("\n") == 3
    assert json.loads((run_dir / "report.json").read_text(encoding="utf-8"))["verdict"]


def test_an_eval_run_is_never_overwritten(tmp_path: Path) -> None:
    report = build_mapping_report(GoldSet(_head(1, "sam")), _decisions(1))
    args = dict(
        run_id="run-1", report=report, subject=_SUBJECT, decisions=_decisions(1),
        gold_rows_sha256="a", universe_sha256="b",
    )
    write_eval_run(tmp_path, **args)  # type: ignore[arg-type]

    with pytest.raises(FileExistsError):
        write_eval_run(tmp_path, **args)  # type: ignore[arg-type]


def test_the_decision_digest_is_order_independent(tmp_path: Path) -> None:
    report = build_mapping_report(GoldSet(_head(3, "sam")), _decisions(3))
    common = dict(report=report, subject=_SUBJECT, gold_rows_sha256="a", universe_sha256="b")

    forward = write_eval_run(tmp_path, run_id="a", decisions=_decisions(3), **common)  # type: ignore[arg-type]
    reverse = write_eval_run(
        tmp_path, run_id="b", decisions=list(reversed(_decisions(3))), **common  # type: ignore[arg-type]
    )

    assert json.loads((forward / "run.json").read_text())["decisions_sha256"] == json.loads(
        (reverse / "run.json").read_text()
    )["decisions_sha256"]
