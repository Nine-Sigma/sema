"""Unit-level tests for the planner storage layout (round-trip in memory)."""

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

    return Provenance(
        run=RunProvenance(
            run_id="run-1",
            target_model_version="omop-cdm-5.4",
            target_schema_snapshot_hash="t",
            vocab_release="v",
            context_card_version="c",
            prompt_template_version="t1",
            few_shot_set_version="f",
            constraint_version="cv",
            llm_model="m",
            embedding_model="e",
        ),
        source=SourceScope(
            source_id="cbioportal_gbm",
            source_schema_hash="s",
            source_profile_hash="p",
        ),
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_planner_migrations_emit_required_constraints() -> None:
    from sema.graph.planner_migrations import cypher_up

    statements = cypher_up(enterprise=True)
    assert any("MappingAssertion" in s for s in statements)
    assert any("HumanPin" in s for s in statements)
    assert any("model_role IS NOT NULL" in s for s in statements)
    assert any("prov_run_run_id" in s for s in statements)


def test_community_migration_omits_existence_constraints() -> None:
    from sema.graph.planner_migrations import cypher_up

    statements = cypher_up(enterprise=False)
    assert not any("model_role IS NOT NULL" in s for s in statements)
    assert any("MappingAssertion_id_unique" in s for s in statements)


def test_planner_migrations_down_drops_planner_data() -> None:
    from sema.graph.planner_migrations import cypher_down

    statements = cypher_down()
    assert any("DETACH DELETE" in s for s in statements)
    assert any("MappingAssertion" in s for s in statements)


def test_apoc_triggers_emitted_when_requested() -> None:
    from sema.graph.planner_migrations import cypher_up

    statements = cypher_up(apoc=True)
    triggers = [s for s in statements if "apoc.trigger.add" in s]
    assert len(triggers) == 7
    assert any("'MAPS_TO'" in s and "'TARGET'" in s for s in triggers)
    assert any("'DERIVED_FROM'" in s and "'SOURCE'" in s for s in triggers)
    assert any("'HAS_LINEAGE'" in s and "'SOURCE'" in s for s in triggers)
    assert any(
        "'RESOLUTION_INPUT'" in s and "'SOURCE'" in s for s in triggers
    )
    assert any("planner_no_role_id_collision" in s for s in triggers)
    assert any("planner_source_role_requires_source_id" in s for s in triggers)
    assert any(
        "planner_target_role_requires_target_model_id" in s for s in triggers
    )


def test_role_backfill_includes_source_id_derivation() -> None:
    from sema.graph.planner_migrations import cypher_up

    statements = cypher_up(enterprise=False)
    backfill_stmts = [s for s in statements if "model_role" in s or "source_id" in s]
    assert any("source_id IS NULL" in s for s in backfill_stmts)
    assert any("source_schema" in s for s in backfill_stmts)
    assert any("model_role = 'SOURCE'" in s for s in backfill_stmts)


def test_apoc_triggers_omitted_by_default() -> None:
    from sema.graph.planner_migrations import cypher_up

    assert not any(
        "apoc.trigger.add" in s for s in cypher_up(enterprise=False)
    )


def test_cypher_down_apoc_true_removes_triggers() -> None:
    from sema.graph.planner_migrations import cypher_down

    statements = cypher_down(apoc=True)
    assert any(
        "apoc.trigger.remove('planner_maps_to_requires_target_property')" in s
        for s in statements
    )


def test_cypher_down_apoc_false_skips_trigger_removal() -> None:
    from sema.graph.planner_migrations import cypher_down

    statements = cypher_down(apoc=False)
    assert not any("apoc.trigger.remove" in s for s in statements)
    assert any("MappingAssertion" in s for s in statements)


def test_provenance_round_trip_via_native_properties() -> None:
    from sema.graph.planner_loader import (
        properties_to_provenance,
        provenance_to_properties,
    )

    prov = _provenance()
    props = provenance_to_properties(prov)
    assert props["prov_run_run_id"] == "run-1"
    assert props["prov_source_source_id"] == "cbioportal_gbm"
    assert props["prov_timestamp"].startswith("2026-01-01")
    rt = properties_to_provenance(props)
    assert rt.run.run_id == prov.run.run_id
    assert rt.source.source_profile_hash == prov.source.source_profile_hash


def test_confirmed_under_round_trip() -> None:
    from sema.graph.planner_loader import (
        confirmed_under_to_properties,
        properties_to_confirmed_under,
    )

    prov = _provenance()
    props = confirmed_under_to_properties(prov.run, prov.source)
    rt_run, rt_source = properties_to_confirmed_under(props)
    assert rt_run == prov.run
    assert rt_source == prov.source


def test_maps_to_requires_target_property() -> None:
    from sema.graph.planner_loader import cypher_create_field_map_maps_to
    from sema.models.planner._enums import ModelRole

    stmt, params = cypher_create_field_map_maps_to(
        "fm-1", "p-1", ModelRole.TARGET
    )
    assert "MAPS_TO" in stmt
    assert params == {"fm_id": "fm-1", "p_id": "p-1"}
    with pytest.raises(ValueError, match="MAPS_TO"):
        cypher_create_field_map_maps_to("fm-1", "p-1", ModelRole.SOURCE)


def test_derived_from_requires_source_property() -> None:
    from sema.graph.planner_loader import cypher_create_field_map_derived_from
    from sema.models.planner._enums import ModelRole

    stmt, _ = cypher_create_field_map_derived_from(
        "fm-1", "p-1", ModelRole.SOURCE
    )
    assert "DERIVED_FROM" in stmt
    with pytest.raises(ValueError, match="DERIVED_FROM"):
        cypher_create_field_map_derived_from("fm-1", "p-1", ModelRole.TARGET)


def test_has_lineage_requires_source_property() -> None:
    from sema.graph.planner_loader import cypher_create_plan_has_lineage
    from sema.models.planner._enums import ModelRole

    stmt, _ = cypher_create_plan_has_lineage("plan-1", "p-1", ModelRole.SOURCE)
    assert "HAS_LINEAGE" in stmt
    with pytest.raises(ValueError, match="HAS_LINEAGE"):
        cypher_create_plan_has_lineage("plan-1", "p-1", ModelRole.TARGET)


def test_resolution_input_requires_source_property() -> None:
    from sema.graph.planner_loader import cypher_create_resolution_input
    from sema.models.planner._enums import ModelRole

    stmt, _ = cypher_create_resolution_input("rp-1", "p-1", ModelRole.SOURCE)
    assert "RESOLUTION_INPUT" in stmt
    with pytest.raises(ValueError, match="RESOLUTION_INPUT"):
        cypher_create_resolution_input("rp-1", "p-1", ModelRole.TARGET)


def test_required_property_role_lookup_unknown_returns_none() -> None:
    from sema.models.planner._role_validation import required_property_role

    assert required_property_role("ASSEMBLED_INTO") is None
    assert required_property_role("MAPS_TO") is not None


def _mapping_assertion(**overrides):
    from sema.models.planner.mapping_plan import MappingAssertion
    from sema.models.planner.patterns import (
        DirectCopyPayload,
        MappingPattern,
    )
    from sema.models.planner.lifecycle import Status

    base = dict(
        id="a-1",
        source_field_ref="cbio.gender",
        target_property_ref="omop.gender_concept_id",
        pattern=MappingPattern.DIRECT_COPY,
        payload=DirectCopyPayload(source_field_ref="cbio.gender"),
        confidence=0.92,
        provenance=_provenance(),
        status=Status.candidate,
    )
    base.update(overrides)
    return MappingAssertion(**base)


def test_mapping_assertion_round_trip_via_properties() -> None:
    from sema.graph.planner_loader import (
        mapping_assertion_to_properties,
        properties_to_mapping_assertion,
    )

    a = _mapping_assertion()
    rt = properties_to_mapping_assertion(mapping_assertion_to_properties(a))
    assert rt == a


def test_field_map_round_trip_via_properties() -> None:
    from sema.graph.planner_loader import (
        field_map_to_properties,
        properties_to_field_map,
    )
    from sema.models.planner.field_map import FieldMap
    from sema.models.planner.patterns import DirectCopyPayload, MappingPattern

    fm = FieldMap(
        target_field_ref="omop.person.gender_concept_id",
        pattern=MappingPattern.DIRECT_COPY,
        payload=DirectCopyPayload(source_field_ref="cbio.gender"),
    )
    rt = properties_to_field_map(field_map_to_properties(fm))
    assert rt == fm


def test_target_obligation_round_trip_via_properties() -> None:
    from sema.graph.planner_loader import (
        properties_to_target_obligation,
        target_obligation_to_properties,
    )
    from sema.models.planner._enums import PrimaryKeyStrategy
    from sema.models.planner.target_model import TargetObligation

    o = TargetObligation(
        target_entity="omop.person",
        required_fields=["person_id"],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
    )
    rt = properties_to_target_obligation(target_obligation_to_properties(o))
    assert rt == o


def test_mapping_plan_round_trip_via_properties() -> None:
    from sema.graph.planner_loader import (
        mapping_plan_to_properties,
        properties_to_mapping_plan,
    )
    from sema.models.planner._enums import (
        MaterializationMode,
        PrimaryKeyStrategy,
    )
    from sema.models.planner.field_map import FieldMap, RowIdentity
    from sema.models.planner.mapping_plan import MappingPlan
    from sema.models.planner.patterns import DirectCopyPayload, MappingPattern
    from sema.models.planner.target_model import TargetObligation

    plan = MappingPlan(
        id="plan-1",
        source_scope_ref="cbio",
        obligation=TargetObligation(
            target_entity="omop.person",
            required_fields=["gender_concept_id"],
            primary_key=PrimaryKeyStrategy.NATURAL_KEY,
        ),
        row_identity=RowIdentity(
            target_row_key_rule="hash",
            source_lineage=["cbio.patient_id"],
            materialization_mode=MaterializationMode.MERGE,
        ),
        field_maps=[
            FieldMap(
                target_field_ref="omop.person.gender_concept_id",
                pattern=MappingPattern.DIRECT_COPY,
                payload=DirectCopyPayload(source_field_ref="cbio.gender"),
            )
        ],
        lineage=["cbio.patient", "cbio.gender"],
    )
    rt = properties_to_mapping_plan(mapping_plan_to_properties(plan))
    assert rt == plan


def test_resolution_plan_round_trip_via_properties() -> None:
    from datetime import datetime, timezone

    from sema.graph.planner_loader import (
        properties_to_resolution_plan,
        resolution_plan_to_properties,
    )
    from sema.models.planner.provenance import SourceScope
    from sema.models.planner.resolution import (
        DeterministicHashPayload,
        ResolutionPlan,
        ResolutionStrategy,
    )

    rp = ResolutionPlan(
        id="rp-1",
        sources=[
            SourceScope(
                source_id="cbio", source_schema_hash="s", source_profile_hash="p"
            )
        ],
        target_identity_ref="canonical.patient_id",
        strategy=ResolutionStrategy.DETERMINISTIC_HASH,
        payload=DeterministicHashPayload(source_key_refs=["cbio.patient_id"]),
        confidence=1.0,
        provenance_run=_provenance().run,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    props = resolution_plan_to_properties(rp)
    assert props["prov_run_run_id"] == "run-1"
    assert props["strategy"] == "DETERMINISTIC_HASH"
    rt = properties_to_resolution_plan(props)
    assert rt == rp


def test_risk_flag_round_trip_via_properties() -> None:
    from sema.graph.planner_loader import (
        properties_to_risk_flag,
        risk_flag_to_properties,
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

    rf = RiskFlag(
        code=RiskCode.RISK_VOCAB_DOMAIN_MISMATCH,
        severity=Severity.warn,
        evidence=[
            Evidence(
                mode=EvidenceMode.COUNT_ONLY,
                payload={"count": 3},
                sensitivity_class=SensitivityClass.PHI,
                source_ref="cbio.x",
            )
        ],
        source_stage=SourceStage.producer,
        suggested_action=SuggestedAction.review,
    )
    rt = properties_to_risk_flag(risk_flag_to_properties(rf))
    assert rt == rf


def test_human_pin_round_trip_via_properties_assertion_pin() -> None:
    from datetime import datetime, timezone

    from sema.graph.planner_loader import (
        human_pin_to_properties,
        properties_to_human_pin,
    )
    from sema.models.planner.lifecycle import HumanPin, PinState

    pin = HumanPin(
        pin_id="pin-a1",
        assertion_id="a-1",
        pinned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        pinned_by="reviewer@x",
        confirmed_under_run=_provenance().run,
        confirmed_under_source=_provenance().source,
        pin_state=PinState.active,
    )
    rt = properties_to_human_pin(human_pin_to_properties(pin))
    assert rt == pin


class _FakeRow(dict):
    pass


class _FakeResult:
    def __init__(self, props: dict | None) -> None:
        self._props = props

    def single(self) -> _FakeRow | None:
        if self._props is None:
            return None
        return _FakeRow(p=self._props)


class _FakeSession:
    """Minimal in-memory session double for write/read helper unit tests."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], dict] = {}

    def run(self, cypher: str, **params):
        if "MERGE" in cypher and "SET n = $props" in cypher:
            label = cypher.split(":", 1)[1].split(" ")[0]
            props = params["props"]
            self.store[(label, props["id"])] = dict(props)
            return _FakeResult(None)
        if "MATCH" in cypher and "RETURN properties(n) AS p" in cypher:
            label = cypher.split(":", 1)[1].split(" ")[0]
            node_id = params["id"]
            return _FakeResult(self.store.get((label, node_id)))
        return _FakeResult(None)


def test_write_read_mapping_assertion_via_helpers() -> None:
    from sema.graph.planner_loader import (
        read_mapping_assertion,
        write_mapping_assertion,
    )

    session = _FakeSession()
    a = _mapping_assertion()
    write_mapping_assertion(session, a)
    rt = read_mapping_assertion(session, a.id)
    assert rt == a


def test_read_mapping_assertion_missing_raises() -> None:
    from sema.graph.planner_loader import read_mapping_assertion

    with pytest.raises(LookupError):
        read_mapping_assertion(_FakeSession(), "nope")


def test_write_read_human_pin_via_helpers() -> None:
    from datetime import datetime, timezone

    from sema.graph.planner_loader import read_human_pin, write_human_pin
    from sema.models.planner.lifecycle import HumanPin, PinState

    session = _FakeSession()
    pin = HumanPin(
        pin_id="pin-w",
        assertion_id="a-1",
        pinned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        pinned_by="reviewer@x",
        confirmed_under_run=_provenance().run,
        confirmed_under_source=_provenance().source,
        pin_state=PinState.active,
    )
    write_human_pin(session, pin)
    rt = read_human_pin(session, "pin-w")
    assert rt == pin


def test_write_read_target_obligation_via_helpers() -> None:
    from sema.graph.planner_loader import (
        read_target_obligation,
        write_target_obligation,
    )
    from sema.models.planner._enums import PrimaryKeyStrategy
    from sema.models.planner.target_model import TargetObligation

    session = _FakeSession()
    o = TargetObligation(
        target_entity="omop.person",
        required_fields=["omop.person.person_id"],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
    )
    write_target_obligation(session, o, obligation_id="ob-1")
    rt = read_target_obligation(session, "ob-1")
    assert rt == o


def test_write_read_field_map_via_helpers() -> None:
    from sema.graph.planner_loader import read_field_map, write_field_map
    from sema.models.planner.field_map import FieldMap
    from sema.models.planner.patterns import (
        DirectCopyPayload,
        MappingPattern,
    )

    session = _FakeSession()
    fm = FieldMap(
        target_field_ref="omop.person.gender_concept_id",
        pattern=MappingPattern.DIRECT_COPY,
        payload=DirectCopyPayload(source_field_ref="cbio.gender"),
    )
    write_field_map(session, fm, field_map_id="fm-1")
    rt = read_field_map(session, "fm-1")
    assert rt == fm


def test_write_read_mapping_plan_via_helpers() -> None:
    from sema.graph.planner_loader import read_mapping_plan, write_mapping_plan
    from sema.models.planner._enums import (
        MaterializationMode,
        PrimaryKeyStrategy,
    )
    from sema.models.planner.field_map import FieldMap, RowIdentity
    from sema.models.planner.mapping_plan import MappingPlan
    from sema.models.planner.patterns import (
        DirectCopyPayload,
        MappingPattern,
    )
    from sema.models.planner.target_model import TargetObligation

    plan = MappingPlan(
        id="plan-w",
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
    )
    session = _FakeSession()
    write_mapping_plan(session, plan)
    rt = read_mapping_plan(session, plan.id)
    assert rt == plan


def test_write_read_resolution_plan_via_helpers() -> None:
    from datetime import datetime, timezone

    from sema.graph.planner_loader import (
        read_resolution_plan,
        write_resolution_plan,
    )
    from sema.models.planner.provenance import SourceScope
    from sema.models.planner.resolution import (
        DeterministicHashPayload,
        ResolutionPlan,
        ResolutionStrategy,
    )

    rp = ResolutionPlan(
        id="rp-w",
        sources=[
            SourceScope(
                source_id="cbio",
                source_schema_hash="s",
                source_profile_hash="p",
            )
        ],
        target_identity_ref="canonical.patient_id",
        strategy=ResolutionStrategy.DETERMINISTIC_HASH,
        payload=DeterministicHashPayload(source_key_refs=["cbio.patient_id"]),
        confidence=1.0,
        provenance_run=_provenance().run,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    session = _FakeSession()
    write_resolution_plan(session, rp)
    rt = read_resolution_plan(session, rp.id)
    assert rt == rp


def test_write_read_risk_flag_via_helpers() -> None:
    from sema.graph.planner_loader import read_risk_flag, write_risk_flag
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

    rf = RiskFlag(
        code=RiskCode.RISK_OBLIGATION_FK_UNSATISFIED,
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
    session = _FakeSession()
    write_risk_flag(session, rf, flag_id="rf-w")
    rt = read_risk_flag(session, "rf-w")
    assert rt == rf


def test_read_helpers_raise_on_missing() -> None:
    from sema.graph.planner_loader import (
        read_field_map,
        read_human_pin,
        read_mapping_plan,
        read_resolution_plan,
        read_risk_flag,
        read_target_obligation,
    )

    s = _FakeSession()
    for fn in (
        read_field_map,
        read_human_pin,
        read_mapping_plan,
        read_resolution_plan,
        read_risk_flag,
        read_target_obligation,
    ):
        with pytest.raises(LookupError):
            fn(s, "missing-id")


def test_human_pin_round_trip_via_properties_resolution_pin() -> None:
    from datetime import datetime, timezone

    from sema.graph.planner_loader import (
        human_pin_to_properties,
        properties_to_human_pin,
    )
    from sema.models.planner.lifecycle import HumanPin, PinState

    pin = HumanPin(
        pin_id="pin-r1",
        resolution_plan_id="rp-1",
        pinned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        pinned_by="reviewer@x",
        confirmed_under_run=_provenance().run,
        confirmed_under_source=_provenance().source,
        pin_state=PinState.stale,
    )
    rt = properties_to_human_pin(human_pin_to_properties(pin))
    assert rt == pin
