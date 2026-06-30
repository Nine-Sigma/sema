"""US-008: Slice-0 staging obligation + concrete PlanAssembler.

Covers §1.5(d) status flow (MappingAssertion.status -> FieldMap.status ->
derive_verdict) and §1.5(e) coverage rule (only an accepted, value-producing
FieldMap covers a required obligation field; NO_MAP and CONSTANT(NULL) never do).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.unit


def _provenance() -> object:
    from sema.models.planner.provenance import (
        Provenance,
        RunProvenance,
        SourceScope,
    )

    run = RunProvenance(
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
    source = SourceScope(
        source_id="cbioportal_gbm",
        source_schema_hash="s-abc",
        source_profile_hash="p-abc",
    )
    return Provenance(
        run=run,
        source=source,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _vocab_lookup_assertion(target_ref: str, *, status: object, aid: str = "a-cc"):
    from sema.models.planner.lifecycle import Status
    from sema.models.planner.mapping_plan import MappingAssertion
    from sema.models.planner.patterns import MappingPattern, VocabLookup

    return MappingAssertion(
        id=aid,
        source_field_ref="cbio.sample.oncotree_code",
        target_property_ref=target_ref,
        pattern=MappingPattern.VOCAB_LOOKUP,
        payload=VocabLookup(
            vocabulary_ref="omop.vocab.snomed",
            source_value_ref="cbio.sample.oncotree_code",
            domain_constraint_ref="target.x.domain=Condition",
            require_standard=True,
            allow_zero_default=False,
            resolver_policy_ref="omop.oncotree_to_snomed_condition",
        ),
        confidence=1.0,
        provenance=_provenance(),
        status=status if status is not None else Status.candidate,
    )


def _constant_assertion(target_ref: str, value: object, *, status: object, aid: str):
    from sema.models.planner.lifecycle import Status
    from sema.models.planner.mapping_plan import MappingAssertion
    from sema.models.planner.patterns import ConstantValue, MappingPattern

    return MappingAssertion(
        id=aid,
        source_field_ref="cbio.run.const",
        target_property_ref=target_ref,
        pattern=MappingPattern.CONSTANT,
        payload=ConstantValue(literal_value=value, target_type="string"),
        confidence=1.0,
        provenance=_provenance(),
        status=status if status is not None else Status.candidate,
    )


def _row_identity():
    from sema.models.planner._enums import MaterializationMode
    from sema.models.planner.field_map import RowIdentity

    return RowIdentity(
        target_row_key_rule="hash",
        source_lineage=["cbio.sample.oncotree_code"],
        materialization_mode=MaterializationMode.REPLACE_PARTITION,
    )


def _full_assertions(status_for_cc: object):
    """All three staging required fields covered (condition_concept_id resolved)."""
    from sema.resolve.policies.omop import make_slice0_staging_obligation

    obligation = make_slice0_staging_obligation()
    cc_ref, policy_ref, release_ref = obligation.required_fields
    return obligation, [
        _vocab_lookup_assertion(cc_ref, status=status_for_cc, aid="a-cc"),
        _constant_assertion(
            policy_ref, "omop.oncotree_to_snomed_condition", status=status_for_cc, aid="a-pol"
        ),
        _constant_assertion(release_ref, "omop-2026-q1", status=status_for_cc, aid="a-rel"),
    ]


# --- §1.5(d): FieldMap.status -------------------------------------------------


def test_field_map_status_defaults_to_candidate() -> None:
    from sema.models.planner.field_map import FieldMap
    from sema.models.planner.lifecycle import Status
    from sema.models.planner.patterns import DirectCopyPayload, MappingPattern

    fm = FieldMap(
        target_field_ref="omop.person.person_id",
        pattern=MappingPattern.DIRECT_COPY,
        payload=DirectCopyPayload(source_field_ref="cbio.patient.patient_id"),
    )
    assert fm.status is Status.candidate


def test_field_map_status_round_trips_via_properties() -> None:
    from sema.graph.planner_loader import (
        field_map_to_properties,
        properties_to_field_map,
    )
    from sema.models.planner.field_map import FieldMap
    from sema.models.planner.lifecycle import Status
    from sema.models.planner.patterns import DirectCopyPayload, MappingPattern

    fm = FieldMap(
        target_field_ref="omop.person.person_id",
        pattern=MappingPattern.DIRECT_COPY,
        payload=DirectCopyPayload(source_field_ref="cbio.patient.patient_id"),
        status=Status.auto_accepted,
    )
    props = field_map_to_properties(fm)
    assert props["status"] == Status.auto_accepted.value
    assert properties_to_field_map(props) == fm


# --- §1.5(e): coverage rule ---------------------------------------------------


def _plan_with_field_map(fm) -> object:
    from sema.models.planner._enums import PrimaryKeyStrategy
    from sema.models.planner.mapping_plan import MappingPlan
    from sema.models.planner.target_model import TargetObligation

    obligation = TargetObligation(
        target_entity="t.staging",
        required_fields=[fm.target_field_ref],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
    )
    return MappingPlan(
        id="p-cov",
        source_scope_ref="cbio",
        obligation=obligation,
        row_identity=_row_identity(),
        field_maps=[fm],
        lineage=["cbio.sample.oncotree_code"],
    )


def test_no_map_field_map_does_not_cover_required_field() -> None:
    from sema.models.planner.field_map import FieldMap
    from sema.models.planner.lifecycle import PlanVerdict
    from sema.models.planner.patterns import MappingPattern, NoMapPayload, NoMapScope

    fm = FieldMap(
        target_field_ref="t.staging.concept",
        pattern=MappingPattern.NO_MAP,
        payload=NoMapPayload(
            reason="no standard target",
            scope=NoMapScope.TARGET_PROPERTY,
            target_property_ref="t.staging.concept",
        ),
    )
    assert not fm.covers_required_field()
    assert _plan_with_field_map(fm).covered_required_fields() == set()
    assert _plan_with_field_map(fm).derive_verdict() == PlanVerdict.blocked_by_obligation


def test_constant_null_field_map_does_not_cover_required_field() -> None:
    from sema.models.planner.field_map import FieldMap
    from sema.models.planner.lifecycle import PlanVerdict
    from sema.models.planner.patterns import ConstantValue, MappingPattern

    fm = FieldMap(
        target_field_ref="t.staging.concept",
        pattern=MappingPattern.CONSTANT,
        payload=ConstantValue(literal_value=None, target_type="int"),
    )
    assert not fm.covers_required_field()
    assert _plan_with_field_map(fm).derive_verdict() == PlanVerdict.blocked_by_obligation


def test_constant_non_null_field_map_covers_required_field() -> None:
    from sema.models.planner.field_map import FieldMap
    from sema.models.planner.lifecycle import PlanVerdict
    from sema.models.planner.patterns import ConstantValue, MappingPattern

    fm = FieldMap(
        target_field_ref="t.staging.concept",
        pattern=MappingPattern.CONSTANT,
        payload=ConstantValue(literal_value="x", target_type="string"),
    )
    assert fm.covers_required_field()
    assert _plan_with_field_map(fm).derive_verdict() == PlanVerdict.compilable


def test_rejected_field_map_does_not_cover_required_field() -> None:
    from sema.models.planner.field_map import FieldMap
    from sema.models.planner.lifecycle import Status
    from sema.models.planner.patterns import DirectCopyPayload, MappingPattern

    fm = FieldMap(
        target_field_ref="t.staging.concept",
        pattern=MappingPattern.DIRECT_COPY,
        payload=DirectCopyPayload(source_field_ref="cbio.x.y"),
        status=Status.rejected,
    )
    assert not fm.covers_required_field()


def test_derive_verdict_awaiting_review_when_field_map_review_pending() -> None:
    from sema.models.planner.field_map import FieldMap
    from sema.models.planner.lifecycle import PlanVerdict, Status
    from sema.models.planner.patterns import DirectCopyPayload, MappingPattern

    fm = FieldMap(
        target_field_ref="t.staging.concept",
        pattern=MappingPattern.DIRECT_COPY,
        payload=DirectCopyPayload(source_field_ref="cbio.x.y"),
        status=Status.review_pending,
    )
    assert _plan_with_field_map(fm).derive_verdict() == PlanVerdict.awaiting_review


# --- Slice-0 staging obligation ----------------------------------------------


def test_staging_obligation_three_required_fields_not_nullable() -> None:
    from sema.resolve.policies.omop import make_slice0_staging_obligation

    obligation = make_slice0_staging_obligation()
    fields = obligation.required_fields
    assert len(fields) == 3
    assert any(f.endswith(".condition_concept_id") for f in fields)
    assert any(f.endswith(".resolver_policy_ref") for f in fields)
    assert any(f.endswith(".vocab_release") for f in fields)
    # §1.5(e): condition_concept_id stays REQUIRED, not nullable.
    assert obligation.nullable_fields == []
    # distinct from production condition_occurrence: no person_id, no dates.
    assert not any("person_id" in f or "date" in f for f in fields)


# --- concrete PlanAssembler ---------------------------------------------------


def test_assemble_resolved_assertion_yields_pass() -> None:
    from sema.models.planner.lifecycle import PlanVerdict, Status
    from sema.resolve.assembler import Slice0PlanAssembler

    obligation, assertions = _full_assertions(Status.auto_accepted)
    plan = Slice0PlanAssembler().assemble(assertions, obligation, _row_identity())
    assert plan.derive_verdict() == PlanVerdict.compilable
    assert plan.covered_required_fields() == set(obligation.required_fields)


def test_assemble_missing_required_field_yields_fail() -> None:
    from sema.models.planner.lifecycle import PlanVerdict, Status
    from sema.resolve.assembler import Slice0PlanAssembler

    obligation, assertions = _full_assertions(Status.auto_accepted)
    plan = Slice0PlanAssembler().assemble(assertions[:-1], obligation, _row_identity())
    assert plan.derive_verdict() == PlanVerdict.blocked_by_obligation


def test_assemble_constant_null_required_field_yields_fail() -> None:
    from sema.models.planner.lifecycle import PlanVerdict, Status
    from sema.resolve.assembler import Slice0PlanAssembler
    from sema.resolve.policies.omop import make_slice0_staging_obligation

    obligation = make_slice0_staging_obligation()
    cc_ref, policy_ref, release_ref = obligation.required_fields
    assertions = [
        _vocab_lookup_assertion(cc_ref, status=Status.auto_accepted),
        _constant_assertion(policy_ref, None, status=Status.auto_accepted, aid="a-pol"),
        _constant_assertion(release_ref, "r", status=Status.auto_accepted, aid="a-rel"),
    ]
    plan = Slice0PlanAssembler().assemble(assertions, obligation, _row_identity())
    assert plan.derive_verdict() == PlanVerdict.blocked_by_obligation


def test_assemble_no_map_field_does_not_cover() -> None:
    from sema.models.planner.lifecycle import PlanVerdict, Status
    from sema.models.planner.mapping_plan import MappingAssertion
    from sema.models.planner.patterns import MappingPattern, NoMapPayload, NoMapScope
    from sema.resolve.assembler import Slice0PlanAssembler
    from sema.resolve.policies.omop import make_slice0_staging_obligation

    obligation = make_slice0_staging_obligation()
    cc_ref, policy_ref, release_ref = obligation.required_fields
    no_map = MappingAssertion(
        id="a-nomap",
        source_field_ref="cbio.sample.oncotree_code",
        target_property_ref=cc_ref,
        pattern=MappingPattern.NO_MAP,
        payload=NoMapPayload(
            reason="dead end", scope=NoMapScope.TARGET_PROPERTY, target_property_ref=cc_ref
        ),
        confidence=1.0,
        provenance=_provenance(),
        status=Status.auto_accepted,
    )
    assertions = [
        no_map,
        _constant_assertion(policy_ref, "p", status=Status.auto_accepted, aid="a-pol"),
        _constant_assertion(release_ref, "r", status=Status.auto_accepted, aid="a-rel"),
    ]
    plan = Slice0PlanAssembler().assemble(assertions, obligation, _row_identity())
    assert cc_ref not in plan.covered_required_fields()
    assert plan.derive_verdict() == PlanVerdict.blocked_by_obligation


def test_assemble_sets_field_map_status_from_winning_assertion() -> None:
    from sema.models.planner.lifecycle import Status
    from sema.resolve.assembler import Slice0PlanAssembler

    obligation, assertions = _full_assertions(Status.review_pending)
    plan = Slice0PlanAssembler().assemble(assertions, obligation, _row_identity())
    assert all(fm.status is Status.review_pending for fm in plan.field_maps)


def test_assemble_uses_planner_select_winner_status_tier() -> None:
    from sema.models.planner.lifecycle import Status
    from sema.resolve.assembler import Slice0PlanAssembler
    from sema.resolve.policies.omop import make_slice0_staging_obligation

    obligation = make_slice0_staging_obligation()
    cc_ref, policy_ref, release_ref = obligation.required_fields
    # Two competing condition_concept_id assertions: pinned (lower conf) must win.
    pinned = _vocab_lookup_assertion(cc_ref, status=Status.human_pinned, aid="a-pin")
    auto = _vocab_lookup_assertion(cc_ref, status=Status.auto_accepted, aid="a-auto")
    assertions = [
        auto,
        pinned,
        _constant_assertion(policy_ref, "p", status=Status.auto_accepted, aid="a-pol"),
        _constant_assertion(release_ref, "r", status=Status.auto_accepted, aid="a-rel"),
    ]
    plan = Slice0PlanAssembler().assemble(assertions, obligation, _row_identity())
    cc_map = [fm for fm in plan.field_maps if fm.target_field_ref == cc_ref]
    assert len(cc_map) == 1
    assert cc_map[0].status is Status.human_pinned


def test_assemble_emits_conflict_resolved_flag_without_blocking() -> None:
    from sema.models.planner.lifecycle import PlanVerdict, Status
    from sema.models.planner.risk import RiskCode
    from sema.resolve.assembler import Slice0PlanAssembler
    from sema.resolve.policies.omop import make_slice0_staging_obligation

    obligation = make_slice0_staging_obligation()
    cc_ref, policy_ref, release_ref = obligation.required_fields
    assertions = [
        _vocab_lookup_assertion(cc_ref, status=Status.auto_accepted, aid="a-1"),
        _vocab_lookup_assertion(cc_ref, status=Status.auto_accepted, aid="a-2"),
        _constant_assertion(policy_ref, "p", status=Status.auto_accepted, aid="a-pol"),
        _constant_assertion(release_ref, "r", status=Status.auto_accepted, aid="a-rel"),
    ]
    plan = Slice0PlanAssembler().assemble(assertions, obligation, _row_identity())
    codes = {rf.code for rf in plan.risk_flags}
    assert RiskCode.RISK_ASSEMBLER_CONFLICT_RESOLVED in codes
    assert plan.derive_verdict() == PlanVerdict.compilable


def test_assemble_rejects_empty_assertions() -> None:
    from sema.resolve.assembler import Slice0PlanAssembler
    from sema.resolve.policies.omop import make_slice0_staging_obligation

    with pytest.raises(ValueError):
        Slice0PlanAssembler().assemble(
            [], make_slice0_staging_obligation(), _row_identity()
        )
