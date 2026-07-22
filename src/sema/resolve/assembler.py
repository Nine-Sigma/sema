"""Slice-0 PlanAssembler: compose MappingAssertions into a MappingPlan.

Generic spine module (R29-scanned): carries no vocabulary- or target-specific
literals. The Slice-0 staging obligation (which DOES name showcase fields) is
authored in the allowlisted policy layer (``resolve/policies/omop.py``) and
passed in, never built here.

Implements the ``PlanAssembler`` Protocol (``models/planner/mapping_plan.py``):
groups assertions by target property, resolves each conflict with the planner
``select_winner`` (emitting ``RISK_ASSEMBLER_CONFLICT_RESOLVED`` when a group
had more than one live assertion), projects each winner into a ``FieldMap``
carrying the winning ``Status`` (§1.5(d)), and returns a ``MappingPlan`` whose
``derive_verdict`` folds obligation coverage (§1.5(e)) and review state into one
verdict.
"""

from __future__ import annotations

from collections import defaultdict

from sema.models.planner.field_map import FieldMap, RowIdentity
from sema.models.planner.lifecycle import Status
from sema.models.planner.mapping_plan import (
    ConflictResolutionPolicy,
    MappingAssertion,
    MappingPlan,
    select_winner,
)
from sema.models.planner.risk import (
    Evidence,
    EvidenceMode,
    RiskCode,
    RiskFlag,
    SensitivityClass,
    Severity,
    SourceStage,
    SuggestedAction,
)
from sema.models.planner.target_model import TargetObligation


class Slice0PlanAssembler:
    """Concrete, vocabulary-agnostic PlanAssembler for the Slice-0 spine."""

    conflict_policy: ConflictResolutionPolicy

    def __init__(
        self, conflict_policy: ConflictResolutionPolicy | None = None
    ) -> None:
        self.conflict_policy = conflict_policy or ConflictResolutionPolicy.default()

    def assemble(
        self,
        assertions: list[MappingAssertion],
        obligation: TargetObligation,
        row_identity: RowIdentity,
    ) -> MappingPlan:
        source_scope_ref = _source_scope_ref(assertions)
        field_maps: list[FieldMap] = []
        risk_flags: list[RiskFlag] = []
        for target_ref, group in _group_by_target(assertions).items():
            live = [a for a in group if a.status is not Status.rejected]
            if not live:
                continue
            field_maps.append(_field_map_from(select_winner(live, self.conflict_policy)))
            if len(live) > 1:
                risk_flags.append(_conflict_resolved_flag(target_ref, len(live)))
        return MappingPlan(
            id=f"plan::{obligation.target_entity}::{source_scope_ref}",
            source_scope_ref=source_scope_ref,
            obligation=obligation,
            row_identity=row_identity,
            field_maps=field_maps,
            risk_flags=risk_flags,
            lineage=_lineage(assertions),
        )


def _group_by_target(
    assertions: list[MappingAssertion],
) -> dict[str, list[MappingAssertion]]:
    groups: dict[str, list[MappingAssertion]] = defaultdict(list)
    for a in assertions:
        groups[a.target_property_ref].append(a)
    return groups


def _field_map_from(winner: MappingAssertion) -> FieldMap:
    return FieldMap(
        target_field_ref=winner.target_property_ref,
        pattern=winner.pattern,
        payload=winner.payload,
        status=winner.status,
    )


def _lineage(assertions: list[MappingAssertion]) -> list[str]:
    return sorted(
        {a.source_field_ref for a in assertions if a.status is not Status.rejected}
    )


def _source_scope_ref(assertions: list[MappingAssertion]) -> str:
    for a in assertions:
        if a.status is not Status.rejected:
            return a.provenance.source.source_id
    raise ValueError("cannot assemble a MappingPlan without a non-rejected assertion")


def _conflict_resolved_flag(target_ref: str, count: int) -> RiskFlag:
    return RiskFlag(
        code=RiskCode.RISK_ASSEMBLER_CONFLICT_RESOLVED,
        severity=Severity.info,
        evidence=[
            Evidence(
                mode=EvidenceMode.COUNT_ONLY,
                payload={"count": count},
                sensitivity_class=SensitivityClass.PUBLIC,
                source_ref=target_ref,
            )
        ],
        source_stage=SourceStage.constraint,
        suggested_action=SuggestedAction.review,
    )
