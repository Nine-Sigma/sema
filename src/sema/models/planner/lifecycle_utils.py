"""Helpers for lifecycle-and-pins: state-machine + pin-staleness logic."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Iterable

from pydantic import BaseModel, Field

from sema.models.planner.provenance import RunProvenance, SourceScope

if TYPE_CHECKING:
    from sema.models.planner.lifecycle import HumanPin, PinState, PlanVerdict, Status


_ALLOWED_STATUS_TRANSITIONS: dict[str, set[str]] = {
    "candidate": {"auto_accepted", "review_pending", "rejected", "human_pinned"},
    "auto_accepted": {"review_pending", "human_pinned", "rejected"},
    "review_pending": {"human_pinned", "rejected", "auto_accepted"},
    "human_pinned": {"rejected"},
    "rejected": set(),
}


def transition_status(current: "Status", target: "Status") -> "Status":
    allowed = _ALLOWED_STATUS_TRANSITIONS[current.value]
    if target.value not in allowed:
        raise ValueError(
            f"disallowed assertion Status transition: {current.value} -> {target.value}"
        )
    return target


def derive_plan_verdict(
    *,
    risk_codes: Iterable[str],
    obligation_required_missing: bool,
    fk_unsatisfied: bool,
    minimum_viable_row_violated: bool,
    any_review_pending: bool,
    any_resolution_dependency_missing: bool,
) -> "PlanVerdict":
    from sema.models.planner.lifecycle import PlanVerdict

    if any_review_pending:
        return PlanVerdict.awaiting_review
    if any_resolution_dependency_missing:
        return PlanVerdict.blocked_by_resolution
    if obligation_required_missing or minimum_viable_row_violated:
        return PlanVerdict.blocked_by_obligation
    if fk_unsatisfied:
        return PlanVerdict.blocked_by_fk
    block_codes = {
        "RISK_PIVOT_CARDINALITY_UNVERIFIED",
        "RISK_VOCAB_DOMAIN_MISMATCH",
    }
    if any(code in block_codes for code in risk_codes):
        return PlanVerdict.blocked_by_constraint
    return PlanVerdict.compilable


def _drift_dimensions(
    pin: "HumanPin", current_run: RunProvenance, current_source: SourceScope
) -> set[str]:
    drifted: set[str] = set()
    for dim in pin.expires_on_change_of:
        prior = _read_dim(pin.confirmed_under_run, pin.confirmed_under_source, dim)
        current = _read_dim(current_run, current_source, dim)
        if prior != current:
            drifted.add(dim)
    return drifted


def _read_dim(run: RunProvenance, source: SourceScope, dim: str) -> object:
    if hasattr(run, dim):
        return getattr(run, dim)
    if hasattr(source, dim):
        return getattr(source, dim)
    raise ValueError(f"unknown provenance dimension: {dim}")


def detect_pin_stale(
    pin: "HumanPin", current_run: RunProvenance, current_source: SourceScope
) -> "HumanPin":
    from sema.models.planner.lifecycle import PinState

    if pin.pin_state in (PinState.invalidated,):
        return pin
    drift = _drift_dimensions(pin, current_run, current_source)
    if drift:
        return pin.model_copy(update={"pin_state": PinState.stale})
    return pin


def revalidate(
    pin: "HumanPin",
    *,
    holds: bool,
    current_run: RunProvenance,
    current_source: SourceScope,
) -> "HumanPin":
    from sema.models.planner.lifecycle import PinState

    if holds:
        return pin.model_copy(
            update={
                "pin_state": PinState.revalidated,
                "confirmed_under_run": current_run,
                "confirmed_under_source": current_source,
            }
        )
    return pin.model_copy(update={"pin_state": PinState.invalidated})


def revoke_pin(pin: "HumanPin") -> "HumanPin":
    from sema.models.planner.lifecycle import PinState

    return pin.model_copy(update={"pin_state": PinState.invalidated})


class DispatchDecision(str, Enum):
    SKIP = "SKIP"
    REVALIDATE = "REVALIDATE"
    DISPATCH = "DISPATCH"


class BuildContext(BaseModel):
    pins: list["HumanPin"] = Field(default_factory=list)
    rejected_pairs: list[tuple[str, str]] = Field(default_factory=list)

    def dispatch_decision(self, pin: "HumanPin") -> DispatchDecision:
        from sema.models.planner.lifecycle import PinState

        if pin.pin_state in (PinState.active, PinState.revalidated):
            return DispatchDecision.SKIP
        if pin.pin_state is PinState.stale:
            return DispatchDecision.REVALIDATE
        return DispatchDecision.DISPATCH

    def is_rejected(self, pair: tuple[str, str]) -> bool:
        return pair in set(self.rejected_pairs)
