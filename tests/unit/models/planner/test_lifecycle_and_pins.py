"""Tests for the lifecycle-and-pins capability."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def _run_prov(**overrides: object) -> object:
    from sema.models.planner.provenance import RunProvenance

    base = dict(
        run_id="run-1",
        target_model_version="omop-cdm-5.4",
        target_schema_snapshot_hash="t-abc",
        vocab_release="omop-2026-q1",
        context_card_version="cards-v3",
        prompt_template_version="tpl-7",
        few_shot_set_version="fs-12",
        constraint_version="rules-v2",
        llm_model="claude-opus-4.7",
        embedding_model="bge-large",
    )
    base.update(overrides)
    return RunProvenance(**base)


def _source(**overrides: object) -> object:
    from sema.models.planner.provenance import SourceScope

    base = dict(
        source_id="cbioportal_gbm",
        source_schema_hash="s-abc",
        source_profile_hash="p-abc",
    )
    base.update(overrides)
    return SourceScope(**base)


def test_status_values() -> None:
    from sema.models.planner.lifecycle import Status

    assert {s.value for s in Status} == {
        "candidate",
        "auto_accepted",
        "review_pending",
        "human_pinned",
        "rejected",
    }


def test_status_transitions_allowed() -> None:
    from sema.models.planner.lifecycle import Status, transition_status

    assert transition_status(Status.candidate, Status.auto_accepted) == Status.auto_accepted
    assert transition_status(Status.candidate, Status.review_pending) == Status.review_pending
    assert transition_status(Status.review_pending, Status.human_pinned) == Status.human_pinned
    assert transition_status(Status.auto_accepted, Status.human_pinned) == Status.human_pinned
    assert transition_status(Status.review_pending, Status.rejected) == Status.rejected


def test_status_transitions_forbidden() -> None:
    from sema.models.planner.lifecycle import Status, transition_status

    with pytest.raises(ValueError):
        transition_status(Status.human_pinned, Status.auto_accepted)
    with pytest.raises(ValueError):
        transition_status(Status.rejected, Status.candidate)


def test_plan_verdict_values() -> None:
    from sema.models.planner.lifecycle import PlanVerdict

    assert {v.value for v in PlanVerdict} == {
        "compilable",
        "blocked_by_obligation",
        "blocked_by_constraint",
        "blocked_by_resolution",
        "blocked_by_fk",
        "awaiting_review",
    }


def test_plan_verdict_compilable_when_clean() -> None:
    from sema.models.planner.lifecycle import PlanVerdict, derive_plan_verdict

    v = derive_plan_verdict(
        risk_codes=[],
        obligation_required_missing=False,
        fk_unsatisfied=False,
        minimum_viable_row_violated=False,
        any_review_pending=False,
        any_resolution_dependency_missing=False,
    )
    assert v == PlanVerdict.compilable


def test_plan_verdict_blocked_by_obligation() -> None:
    from sema.models.planner.lifecycle import PlanVerdict, derive_plan_verdict

    v = derive_plan_verdict(
        risk_codes=["RISK_OBLIGATION_REQUIRED_FIELD_MISSING"],
        obligation_required_missing=True,
        fk_unsatisfied=False,
        minimum_viable_row_violated=False,
        any_review_pending=False,
        any_resolution_dependency_missing=False,
    )
    assert v == PlanVerdict.blocked_by_obligation


def test_plan_verdict_awaiting_review() -> None:
    from sema.models.planner.lifecycle import PlanVerdict, derive_plan_verdict

    v = derive_plan_verdict(
        risk_codes=[],
        obligation_required_missing=False,
        fk_unsatisfied=False,
        minimum_viable_row_violated=False,
        any_review_pending=True,
        any_resolution_dependency_missing=False,
    )
    assert v == PlanVerdict.awaiting_review


def test_pin_state_values() -> None:
    from sema.models.planner.lifecycle import PinState

    assert {s.value for s in PinState} == {
        "active",
        "stale",
        "revalidated",
        "invalidated",
    }


def test_human_pin_default_expires() -> None:
    from sema.models.planner.lifecycle import HumanPin, PinState

    pin = HumanPin(
        pin_id="pin-1",
        assertion_id="a-1",
        pinned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        pinned_by="reviewer@x",
        confirmed_under_run=_run_prov(),
        confirmed_under_source=_source(),
    )
    assert pin.pin_state == PinState.active
    assert "vocab_release" in pin.expires_on_change_of
    assert "source_profile_hash" in pin.expires_on_change_of


def test_human_pin_must_reference_one_target() -> None:
    from sema.models.planner.lifecycle import HumanPin

    with pytest.raises(ValidationError):
        HumanPin(
            pin_id="p",
            pinned_at=datetime.now(timezone.utc),
            pinned_by="x",
            confirmed_under_run=_run_prov(),
            confirmed_under_source=_source(),
        )

    with pytest.raises(ValidationError):
        HumanPin(
            pin_id="p",
            assertion_id="a-1",
            resolution_plan_id="r-1",
            pinned_at=datetime.now(timezone.utc),
            pinned_by="x",
            confirmed_under_run=_run_prov(),
            confirmed_under_source=_source(),
        )


def test_pin_stales_on_tracked_dim_drift() -> None:
    from sema.models.planner.lifecycle import HumanPin, PinState, detect_pin_stale

    pin = HumanPin(
        pin_id="p",
        assertion_id="a-1",
        pinned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        pinned_by="x",
        confirmed_under_run=_run_prov(),
        confirmed_under_source=_source(),
    )
    new_pin = detect_pin_stale(pin, _run_prov(vocab_release="omop-2026-q2"), _source())
    assert new_pin.pin_state == PinState.stale


def test_pin_does_not_stale_on_untracked_dim() -> None:
    from sema.models.planner.lifecycle import HumanPin, PinState, detect_pin_stale

    pin = HumanPin(
        pin_id="p",
        assertion_id="a-1",
        pinned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        pinned_by="x",
        expires_on_change_of=["target_model_version"],
        confirmed_under_run=_run_prov(),
        confirmed_under_source=_source(),
    )
    new_pin = detect_pin_stale(
        pin, _run_prov(prompt_template_version="tpl-99"), _source()
    )
    assert new_pin.pin_state == PinState.active


def test_pin_stales_on_source_profile_drift() -> None:
    from sema.models.planner.lifecycle import HumanPin, PinState, detect_pin_stale

    pin = HumanPin(
        pin_id="p",
        assertion_id="a-1",
        pinned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        pinned_by="x",
        confirmed_under_run=_run_prov(),
        confirmed_under_source=_source(),
    )
    new_pin = detect_pin_stale(
        pin, _run_prov(), _source(source_profile_hash="DIFFERENT")
    )
    assert new_pin.pin_state == PinState.stale


def test_revalidation_success_transitions_to_revalidated() -> None:
    from sema.models.planner.lifecycle import (
        HumanPin,
        PinState,
        revalidate,
    )

    pin = HumanPin(
        pin_id="p",
        assertion_id="a-1",
        pinned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        pinned_by="x",
        confirmed_under_run=_run_prov(),
        confirmed_under_source=_source(),
        pin_state=PinState.stale,
    )
    new_run = _run_prov(vocab_release="omop-2026-q2")
    new_src = _source(source_profile_hash="DIFFERENT")
    revalidated = revalidate(pin, holds=True, current_run=new_run, current_source=new_src)
    assert revalidated.pin_state == PinState.revalidated
    assert revalidated.confirmed_under_run.vocab_release == "omop-2026-q2"


def test_revalidation_failure_transitions_to_invalidated() -> None:
    from sema.models.planner.lifecycle import (
        HumanPin,
        PinState,
        revalidate,
    )

    pin = HumanPin(
        pin_id="p",
        assertion_id="a-1",
        pinned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        pinned_by="x",
        confirmed_under_run=_run_prov(),
        confirmed_under_source=_source(),
        pin_state=PinState.stale,
    )
    invalidated = revalidate(pin, holds=False, current_run=_run_prov(), current_source=_source())
    assert invalidated.pin_state == PinState.invalidated


def test_reviewer_revocation_invalidates() -> None:
    from sema.models.planner.lifecycle import (
        HumanPin,
        PinState,
        revoke_pin,
    )

    pin = HumanPin(
        pin_id="p",
        assertion_id="a-1",
        pinned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        pinned_by="x",
        confirmed_under_run=_run_prov(),
        confirmed_under_source=_source(),
    )
    revoked = revoke_pin(pin)
    assert revoked.pin_state == PinState.invalidated


def test_build_context_dispatch_rules() -> None:
    from sema.models.planner.lifecycle import (
        BuildContext,
        DispatchDecision,
        HumanPin,
        PinState,
    )

    pin_active = HumanPin(
        pin_id="p1",
        assertion_id="a1",
        pinned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        pinned_by="x",
        confirmed_under_run=_run_prov(),
        confirmed_under_source=_source(),
    )
    pin_stale = pin_active.model_copy(update={"pin_id": "p2", "pin_state": PinState.stale})
    pin_invalid = pin_active.model_copy(update={"pin_id": "p3", "pin_state": PinState.invalidated})

    ctx = BuildContext(pins=[pin_active, pin_stale, pin_invalid], rejected_pairs=[("s", "t")])

    assert ctx.dispatch_decision(pin_active) == DispatchDecision.SKIP
    assert ctx.dispatch_decision(pin_stale) == DispatchDecision.REVALIDATE
    assert ctx.dispatch_decision(pin_invalid) == DispatchDecision.DISPATCH
    assert ctx.is_rejected(("s", "t"))
    assert not ctx.is_rejected(("u", "v"))
