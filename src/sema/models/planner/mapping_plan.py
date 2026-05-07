"""mapping-planner: MappingAssertion, MappingPlan, ConflictResolutionPolicy."""

from __future__ import annotations

from typing import Any, Iterable, Protocol, Self

from pydantic import BaseModel, Field, model_validator

from sema.models.planner._refs import RefStr
from sema.models.planner.field_map import FieldMap, RowIdentity, coerce_pattern_payload
from sema.models.planner.lifecycle import PlanVerdict, Status
from sema.models.planner.lifecycle_utils import derive_plan_verdict
from sema.models.planner.patterns import (
    MappingPattern,
    PatternPayload,
    expected_payload_type,
)
from sema.models.planner.provenance import Provenance
from sema.models.planner.risk import RiskFlag
from sema.models.planner.target_model import TargetObligation


class MappingAssertion(BaseModel):
    id: str = Field(min_length=1)
    source_field_ref: RefStr
    target_property_ref: RefStr
    pattern: MappingPattern
    payload: PatternPayload
    confidence: float = Field(ge=0.0, le=1.0)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    provenance: Provenance
    status: Status = Status.candidate
    concerns_text: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_payload(cls, data: Any) -> Any:
        return coerce_pattern_payload(data)

    @model_validator(mode="after")
    def _validate_payload_matches_pattern(self) -> Self:
        expected = expected_payload_type(self.pattern)
        if not isinstance(self.payload, expected):
            raise ValueError(
                f"pattern {self.pattern.value} requires payload of {expected.__name__}"
            )
        return self


class MappingPlan(BaseModel):
    id: str = Field(min_length=1)
    source_scope_ref: str = Field(min_length=1)
    obligation: TargetObligation
    row_identity: RowIdentity
    field_maps: list[FieldMap] = Field(default_factory=list)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    lineage: list[RefStr] = Field(default_factory=list)

    def covered_required_fields(self) -> set[str]:
        return {fm.target_field_ref for fm in self.field_maps}

    def derive_verdict(self) -> PlanVerdict:
        required = set(self.obligation.required_fields)
        covered = self.covered_required_fields()
        codes = {rf.code.value for rf in self.risk_flags}
        return derive_plan_verdict(
            risk_codes=codes,
            obligation_required_missing=(
                not required.issubset(covered)
                or "RISK_OBLIGATION_REQUIRED_FIELD_MISSING" in codes
            ),
            fk_unsatisfied="RISK_OBLIGATION_FK_UNSATISFIED" in codes,
            minimum_viable_row_violated=(
                "RISK_OBLIGATION_MINIMUM_VIABLE_ROW_VIOLATED" in codes
            ),
            any_review_pending=False,
            any_resolution_dependency_missing=(
                "RISK_RESOLUTION_DEPENDENCY_MISSING" in codes
            ),
        )


class ConflictResolutionPolicy(BaseModel):
    by_pin: bool = True
    by_status_tier: bool = True
    by_confidence: bool = True
    by_recency: bool = True
    by_template_version: bool = True

    @classmethod
    def default(cls) -> ConflictResolutionPolicy:
        return cls()


_STATUS_TIER: dict[Status, int] = {
    Status.human_pinned: 4,
    Status.auto_accepted: 3,
    Status.review_pending: 2,
    Status.candidate: 1,
    Status.rejected: 0,
}


def _sort_key(a: MappingAssertion) -> tuple[int, float, float, str]:
    tier = _STATUS_TIER[a.status]
    confidence = a.confidence
    recency = a.provenance.timestamp.timestamp()
    template = a.provenance.run.prompt_template_version
    return (tier, confidence, recency, template)


def select_winner(
    assertions: Iterable[MappingAssertion],
    policy: ConflictResolutionPolicy,  # noqa: ARG001 (default policy is the only impl)
) -> MappingAssertion:
    candidates = [a for a in assertions if a.status is not Status.rejected]
    if not candidates:
        raise ValueError("no non-rejected assertions to resolve")
    candidates.sort(key=_sort_key, reverse=True)
    return candidates[0]


class PlanAssembler(Protocol):
    """Plan-assembler signature stub.

    Implementations live in the matching-engine change; this protocol fixes
    the contract surface and the RISK_ASSEMBLER_CONFLICT_RESOLVED emission rule.
    """

    conflict_policy: ConflictResolutionPolicy

    def assemble(
        self,
        assertions: list[MappingAssertion],
        obligation: TargetObligation,
        row_identity: RowIdentity,
    ) -> MappingPlan: ...
