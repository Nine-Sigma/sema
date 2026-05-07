"""Tests for MappingAssertion / MappingPlan / ConflictResolutionPolicy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def _provenance(ts: datetime | None = None) -> object:
    from sema.models.planner.provenance import (
        Provenance,
        RunProvenance,
        SourceScope,
    )

    rp = RunProvenance(
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
    src = SourceScope(
        source_id="cbioportal_gbm",
        source_schema_hash="s-abc",
        source_profile_hash="p-abc",
    )
    return Provenance(
        run=rp,
        source=src,
        timestamp=ts or datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_assertion(**overrides: object) -> object:
    from sema.models.planner.mapping_plan import MappingAssertion
    from sema.models.planner.patterns import DirectCopyPayload, MappingPattern
    from sema.models.planner.lifecycle import Status

    base: dict[str, object] = dict(
        id="a-1",
        source_field_ref="cbio.patient.gender",
        target_property_ref="omop.person.gender_concept_id",
        pattern=MappingPattern.DIRECT_COPY,
        payload=DirectCopyPayload(source_field_ref="cbio.patient.gender"),
        confidence=0.9,
        risk_flags=[],
        provenance=_provenance(),
        status=Status.candidate,
    )
    base.update(overrides)
    return MappingAssertion(**base)


def test_field_map_requires_matching_payload() -> None:
    from sema.models.planner.field_map import FieldMap
    from sema.models.planner.patterns import (
        ConstantValue,
        DirectCopyPayload,
        MappingPattern,
    )

    fm = FieldMap(
        target_field_ref="omop.person.gender_concept_id",
        pattern=MappingPattern.DIRECT_COPY,
        payload=DirectCopyPayload(source_field_ref="cbio.patient.gender"),
    )
    assert fm.pattern is MappingPattern.DIRECT_COPY

    with pytest.raises(ValidationError):
        FieldMap(
            target_field_ref="omop.person.x",
            pattern=MappingPattern.DIRECT_COPY,
            payload=ConstantValue(literal_value=1, target_type="int"),
        )


def test_row_identity_requires_lineage() -> None:
    from sema.models.planner._enums import MaterializationMode
    from sema.models.planner.field_map import RowIdentity

    with pytest.raises(ValidationError):
        RowIdentity(
            target_row_key_rule="hash",
            source_lineage=[],
            materialization_mode=MaterializationMode.MERGE,
        )


def test_row_identity_stable_under_same_inputs() -> None:
    from sema.models.planner._enums import MaterializationMode
    from sema.models.planner.field_map import RowIdentity, derive_row_key

    ri = RowIdentity(
        target_row_key_rule="hash",
        source_lineage=["cbio.patient.patient_id", "cbio.study_id"],
        materialization_mode=MaterializationMode.MERGE,
    )
    a = derive_row_key(
        ri, {"cbio.patient.patient_id": "P1", "cbio.study_id": "GBM"}
    )
    b = derive_row_key(
        ri, {"cbio.patient.patient_id": "P1", "cbio.study_id": "GBM"}
    )
    c = derive_row_key(
        ri, {"cbio.patient.patient_id": "P2", "cbio.study_id": "GBM"}
    )
    assert a == b
    assert a != c


def test_mapping_assertion_round_trip() -> None:
    a = _make_assertion()
    payload = a.model_dump(mode="json")
    rt = type(a).model_validate(payload)
    assert rt.id == "a-1"
    from sema.models.planner.patterns import DirectCopyPayload

    assert isinstance(rt.payload, DirectCopyPayload)


def test_mapping_assertion_pattern_payload_mismatch_rejected() -> None:
    from sema.models.planner.lifecycle import Status
    from sema.models.planner.mapping_plan import MappingAssertion
    from sema.models.planner.patterns import ConstantValue, MappingPattern

    with pytest.raises(ValidationError):
        MappingAssertion(
            id="a-bad",
            source_field_ref="cbio.x",
            target_property_ref="omop.y",
            pattern=MappingPattern.DIRECT_COPY,
            payload=ConstantValue(literal_value=1, target_type="int"),
            confidence=0.9,
            risk_flags=[],
            provenance=_provenance(),
            status=Status.candidate,
        )


def test_mapping_plan_round_trip() -> None:
    from sema.models.planner._enums import MaterializationMode, PrimaryKeyStrategy
    from sema.models.planner.field_map import FieldMap, RowIdentity
    from sema.models.planner.mapping_plan import MappingPlan
    from sema.models.planner.patterns import DirectCopyPayload, MappingPattern
    from sema.models.planner.target_model import (
        FieldPresence,
        RowPredicate,
        TargetObligation,
    )

    obligation = TargetObligation(
        target_entity="omop.person",
        required_fields=["person_id"],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
        minimum_viable_row=RowPredicate(
            op="AND", clauses=[FieldPresence(field="person_id")]
        ),
    )
    plan = MappingPlan(
        id="plan-1",
        source_scope_ref="cbioportal_gbm",
        obligation=obligation,
        row_identity=RowIdentity(
            target_row_key_rule="hash",
            source_lineage=["cbio.patient.patient_id"],
            materialization_mode=MaterializationMode.MERGE,
        ),
        field_maps=[
            FieldMap(
                target_field_ref="omop.person.person_id",
                pattern=MappingPattern.DIRECT_COPY,
                payload=DirectCopyPayload(source_field_ref="cbio.patient.patient_id"),
            )
        ],
        risk_flags=[],
        lineage=["cbio.patient.patient_id"],
    )
    rt = MappingPlan.model_validate(plan.model_dump(mode="json"))
    assert rt.id == "plan-1"


def test_mapping_plan_blocked_when_required_missing() -> None:
    from sema.models.planner._enums import MaterializationMode, PrimaryKeyStrategy
    from sema.models.planner.field_map import RowIdentity
    from sema.models.planner.lifecycle import PlanVerdict
    from sema.models.planner.mapping_plan import MappingPlan
    from sema.models.planner.target_model import TargetObligation

    obligation = TargetObligation(
        target_entity="omop.person",
        required_fields=["person_id", "gender_concept_id"],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
    )
    plan = MappingPlan(
        id="plan-2",
        source_scope_ref="cbio",
        obligation=obligation,
        row_identity=RowIdentity(
            target_row_key_rule="hash",
            source_lineage=["x.y"],
            materialization_mode=MaterializationMode.MERGE,
        ),
        field_maps=[],
        risk_flags=[],
        lineage=["x.y"],
    )
    assert plan.derive_verdict() == PlanVerdict.blocked_by_obligation


def _compilable_plan_factory():
    from sema.models.planner._enums import MaterializationMode, PrimaryKeyStrategy
    from sema.models.planner.field_map import FieldMap, RowIdentity
    from sema.models.planner.mapping_plan import MappingPlan
    from sema.models.planner.patterns import DirectCopyPayload, MappingPattern
    from sema.models.planner.target_model import TargetObligation

    def make(risk_flags=None):
        return MappingPlan(
            id="plan-rf",
            source_scope_ref="cbio",
            obligation=TargetObligation(
                target_entity="omop.person",
                required_fields=["omop.person.person_id"],
                primary_key=PrimaryKeyStrategy.NATURAL_KEY,
            ),
            row_identity=RowIdentity(
                target_row_key_rule="hash",
                source_lineage=["cbio.x"],
                materialization_mode=MaterializationMode.MERGE,
            ),
            field_maps=[
                FieldMap(
                    target_field_ref="omop.person.person_id",
                    pattern=MappingPattern.DIRECT_COPY,
                    payload=DirectCopyPayload(source_field_ref="cbio.patient_id"),
                )
            ],
            risk_flags=risk_flags or [],
            lineage=["cbio.x"],
        )

    return make


def _risk(code):
    from sema.models.planner.risk import (
        Evidence,
        EvidenceMode,
        RiskFlag,
        SensitivityClass,
        Severity,
        SourceStage,
        SuggestedAction,
    )

    return RiskFlag(
        code=code,
        severity=Severity.block,
        evidence=[
            Evidence(
                mode=EvidenceMode.COUNT_ONLY,
                payload={"count": 1},
                sensitivity_class=SensitivityClass.PUBLIC,
                source_ref="cbio.x",
            )
        ],
        source_stage=SourceStage.constraint,
        suggested_action=SuggestedAction.review,
    )


def test_plan_verdict_blocked_by_fk_when_risk_present() -> None:
    from sema.models.planner.lifecycle import PlanVerdict
    from sema.models.planner.risk import RiskCode

    make = _compilable_plan_factory()
    plan = make(risk_flags=[_risk(RiskCode.RISK_OBLIGATION_FK_UNSATISFIED)])
    assert plan.derive_verdict() == PlanVerdict.blocked_by_fk


def test_plan_verdict_blocked_by_obligation_when_min_viable_row_violated() -> None:
    from sema.models.planner.lifecycle import PlanVerdict
    from sema.models.planner.risk import RiskCode

    make = _compilable_plan_factory()
    plan = make(
        risk_flags=[
            _risk(RiskCode.RISK_OBLIGATION_MINIMUM_VIABLE_ROW_VIOLATED)
        ]
    )
    assert plan.derive_verdict() == PlanVerdict.blocked_by_obligation


def test_plan_verdict_blocked_by_resolution_dependency_missing() -> None:
    from sema.models.planner.lifecycle import PlanVerdict
    from sema.models.planner.risk import RiskCode

    make = _compilable_plan_factory()
    plan = make(risk_flags=[_risk(RiskCode.RISK_RESOLUTION_DEPENDENCY_MISSING)])
    assert plan.derive_verdict() == PlanVerdict.blocked_by_resolution


def test_plan_verdict_compilable_with_no_blocking_risk() -> None:
    from sema.models.planner.lifecycle import PlanVerdict

    make = _compilable_plan_factory()
    assert make().derive_verdict() == PlanVerdict.compilable


def test_plan_verdict_blocked_by_obligation_when_required_field_missing_risk() -> None:
    from sema.models.planner.lifecycle import PlanVerdict
    from sema.models.planner.risk import RiskCode

    make = _compilable_plan_factory()
    plan = make(
        risk_flags=[_risk(RiskCode.RISK_OBLIGATION_REQUIRED_FIELD_MISSING)]
    )
    assert plan.derive_verdict() == PlanVerdict.blocked_by_obligation


def test_conflict_policy_pin_wins() -> None:
    from sema.models.planner.lifecycle import Status
    from sema.models.planner.mapping_plan import (
        ConflictResolutionPolicy,
        select_winner,
    )

    a1 = _make_assertion(id="a1", confidence=0.6, status=Status.human_pinned)
    a2 = _make_assertion(id="a2", confidence=0.99, status=Status.auto_accepted)
    winner = select_winner([a1, a2], ConflictResolutionPolicy.default())
    assert winner.id == "a1"


def test_conflict_policy_confidence_then_recency_then_template() -> None:
    from sema.models.planner.lifecycle import Status
    from sema.models.planner.mapping_plan import (
        ConflictResolutionPolicy,
        select_winner,
    )

    base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    a1 = _make_assertion(id="a1", confidence=0.91, status=Status.auto_accepted)
    a2 = _make_assertion(id="a2", confidence=0.86, status=Status.auto_accepted)
    assert select_winner([a1, a2], ConflictResolutionPolicy.default()).id == "a1"

    a3 = _make_assertion(
        id="a3",
        confidence=0.91,
        status=Status.auto_accepted,
        provenance=_provenance(ts=base_ts),
    )
    a4 = _make_assertion(
        id="a4",
        confidence=0.91,
        status=Status.auto_accepted,
        provenance=_provenance(ts=base_ts + timedelta(seconds=10)),
    )
    assert select_winner([a3, a4], ConflictResolutionPolicy.default()).id == "a4"
