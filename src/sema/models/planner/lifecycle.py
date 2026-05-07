"""lifecycle-and-pins capability."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, model_validator

from sema.models.planner.lifecycle_utils import (
    BuildContext,
    DispatchDecision,
    derive_plan_verdict,
    detect_pin_stale,
    revalidate,
    revoke_pin,
    transition_status,
)
from sema.models.planner.provenance import RunProvenance, SourceScope


class Status(str, Enum):
    candidate = "candidate"
    auto_accepted = "auto_accepted"
    review_pending = "review_pending"
    human_pinned = "human_pinned"
    rejected = "rejected"


class PlanVerdict(str, Enum):
    compilable = "compilable"
    blocked_by_obligation = "blocked_by_obligation"
    blocked_by_constraint = "blocked_by_constraint"
    blocked_by_resolution = "blocked_by_resolution"
    blocked_by_fk = "blocked_by_fk"
    awaiting_review = "awaiting_review"


class PinState(str, Enum):
    active = "active"
    stale = "stale"
    revalidated = "revalidated"
    invalidated = "invalidated"


_DEFAULT_EXPIRES_ON_CHANGE_OF = (
    "target_model_version",
    "target_schema_snapshot_hash",
    "source_schema_hash",
    "source_profile_hash",
    "context_card_version",
    "vocab_release",
)


class HumanPin(BaseModel):
    pin_id: str = Field(min_length=1)
    assertion_id: str | None = None
    resolution_plan_id: str | None = None
    pinned_at: datetime
    pinned_by: str = Field(min_length=1)
    confirmed_under_run: RunProvenance
    confirmed_under_source: SourceScope
    expires_on_change_of: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_EXPIRES_ON_CHANGE_OF)
    )
    pin_state: PinState = PinState.active

    @model_validator(mode="after")
    def _validate_target(self) -> Self:
        a, r = self.assertion_id, self.resolution_plan_id
        if (a is None) == (r is None):
            raise ValueError("HumanPin MUST reference exactly one of assertion_id or resolution_plan_id")
        return self


__all__ = [
    "Status",
    "PlanVerdict",
    "PinState",
    "HumanPin",
    "transition_status",
    "derive_plan_verdict",
    "detect_pin_stale",
    "revalidate",
    "revoke_pin",
    "BuildContext",
    "DispatchDecision",
]
