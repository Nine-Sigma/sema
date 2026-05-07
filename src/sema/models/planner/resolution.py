"""resolution-planner capability."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from sema.models.planner._refs import RefStr
from sema.models.planner.lifecycle import Status
from sema.models.planner.provenance import RunProvenance, SourceScope
from sema.models.planner.risk import RiskFlag


class ResolutionStrategy(str, Enum):
    DETERMINISTIC_HASH = "DETERMINISTIC_HASH"
    FUZZY_BLOCK_AND_SCORE = "FUZZY_BLOCK_AND_SCORE"
    GRAPH_CLOSURE = "GRAPH_CLOSURE"
    MULTI_KEY_UNION = "MULTI_KEY_UNION"


class CycleHandling(str, Enum):
    REJECT = "REJECT"
    BREAK_AT_DEPTH = "BREAK_AT_DEPTH"
    MARK_AND_CONTINUE = "MARK_AND_CONTINUE"


class ResolutionVerdict(str, Enum):
    resolved = "resolved"
    ambiguous = "ambiguous"
    unresolved = "unresolved"
    awaiting_review = "awaiting_review"


class DeterministicHashPayload(BaseModel):
    source_key_refs: list[RefStr] = Field(min_length=1)


class FuzzyBlockAndScorePayload(BaseModel):
    blocking_keys: list[RefStr] = Field(min_length=1)
    similarity_features: list[RefStr] = Field(min_length=1)


class GraphClosurePayload(BaseModel):
    walk_relationship: str = Field(min_length=1)
    max_depth: int | None = Field(default=None, gt=0)


class MultiKeyUnionPayload(BaseModel):
    source_key_refs: list[RefStr] = Field(min_length=2)


ResolutionPayload = (
    DeterministicHashPayload
    | FuzzyBlockAndScorePayload
    | GraphClosurePayload
    | MultiKeyUnionPayload
)


_STRATEGY_PAYLOAD_TYPES: dict[ResolutionStrategy, type[BaseModel]] = {
    ResolutionStrategy.DETERMINISTIC_HASH: DeterministicHashPayload,
    ResolutionStrategy.FUZZY_BLOCK_AND_SCORE: FuzzyBlockAndScorePayload,
    ResolutionStrategy.GRAPH_CLOSURE: GraphClosurePayload,
    ResolutionStrategy.MULTI_KEY_UNION: MultiKeyUnionPayload,
}


class CycleHandlingRule(BaseModel):
    handling: CycleHandling
    depth: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_depth(self) -> Self:
        if self.handling is CycleHandling.BREAK_AT_DEPTH and self.depth is None:
            raise ValueError("BREAK_AT_DEPTH requires depth")
        return self


class ResolutionPlan(BaseModel):
    id: str = Field(min_length=1)
    sources: list[SourceScope] = Field(min_length=1)
    target_identity_ref: RefStr
    strategy: ResolutionStrategy
    payload: ResolutionPayload
    transitive_closure: bool = False
    cycle_handling: CycleHandlingRule | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    provenance_run: RunProvenance
    timestamp: datetime
    status: Status = Status.candidate

    @model_validator(mode="before")
    @classmethod
    def _coerce_payload_type(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        strategy = data.get("strategy")
        payload = data.get("payload")
        if isinstance(payload, dict) and isinstance(strategy, str):
            target = _STRATEGY_PAYLOAD_TYPES.get(ResolutionStrategy(strategy))
            if target is not None:
                data["payload"] = target.model_validate(payload)
        return data

    @model_validator(mode="after")
    def _validate_strategy_invariants(self) -> Self:
        _validate_payload(self.strategy, self.payload)
        _validate_closure(self.strategy, self.transitive_closure, self.cycle_handling)
        _validate_confidence(self.strategy, self.confidence)
        return self


def _validate_payload(
    strategy: ResolutionStrategy, payload: ResolutionPayload
) -> None:
    expected = _STRATEGY_PAYLOAD_TYPES[strategy]
    if not isinstance(payload, expected):
        raise ValueError(
            f"strategy {strategy.value} requires payload of {expected.__name__}"
        )


def _validate_closure(
    strategy: ResolutionStrategy,
    transitive: bool,
    cycle: CycleHandlingRule | None,
) -> None:
    if strategy is ResolutionStrategy.GRAPH_CLOSURE and not transitive:
        raise ValueError("GRAPH_CLOSURE requires transitive_closure=True")
    if transitive and cycle is None:
        raise ValueError("transitive_closure=True requires cycle_handling")
    if not transitive and cycle is not None:
        raise ValueError("cycle_handling rejected when transitive_closure=False")


def _validate_confidence(
    strategy: ResolutionStrategy, confidence: float
) -> None:
    if strategy is ResolutionStrategy.DETERMINISTIC_HASH and confidence != 1.0:
        raise ValueError("DETERMINISTIC_HASH requires confidence=1.0")


def derive_resolution_verdict(
    *,
    produced_for_every_input: bool,
    ambiguous_assignments: bool,
    cycle_blocked: bool,
    any_block_flag: bool,
    plan_review_pending: bool,
) -> ResolutionVerdict:
    if plan_review_pending:
        return ResolutionVerdict.awaiting_review
    if any_block_flag and cycle_blocked:
        return ResolutionVerdict.unresolved
    if ambiguous_assignments:
        return ResolutionVerdict.ambiguous
    if not produced_for_every_input:
        return ResolutionVerdict.unresolved
    if any_block_flag:
        return ResolutionVerdict.awaiting_review
    return ResolutionVerdict.resolved


class ResolutionDependency(BaseModel):
    upstream_plan_id: str = Field(min_length=1)
    canonical_identity_column: RefStr
