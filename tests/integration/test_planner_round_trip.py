"""Round-trip integration tests for the planner contract storage layer.

Covers tasks 8.10-8.15 and 9.1-9.9: persists each planner Pydantic model to
Neo4j via the migration + loader helpers and asserts byte-identity on
structured fields. Skipped automatically without a running Neo4j.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.integration


from sema.graph.planner_loader import (
    confirmed_under_to_properties,
    human_pin_to_properties,
    mapping_assertion_to_properties,
    properties_to_confirmed_under,
    properties_to_human_pin,
    properties_to_mapping_assertion,
    properties_to_provenance,
    provenance_to_properties,
    read_human_pin,
    read_mapping_assertion,
    read_mapping_plan,
    read_resolution_plan,
    write_human_pin,
    write_mapping_assertion,
    write_mapping_plan,
    write_resolution_plan,
)
from sema.graph.planner_migrations import cypher_down, cypher_up
from sema.models.planner._enums import (
    MaterializationMode,
    ModelRole,
    PrimaryKeyStrategy,
)
from sema.models.planner.field_map import FieldMap, RowIdentity
from sema.models.planner.lifecycle import (
    HumanPin,
    PinState,
    PlanVerdict,
    Status,
)
from sema.models.planner.mapping_plan import (
    ConflictResolutionPolicy,
    MappingAssertion,
    MappingPlan,
    select_winner,
)
from sema.models.planner.patterns import (
    DirectCopyPayload,
    MappingPattern,
    VocabLookup,
)
from sema.models.planner.provenance import (
    PromptArtifact,
    Provenance,
    RunProvenance,
    SourceScope,
    derive_cache_key,
)
from sema.models.planner.resolution import (
    DeterministicHashPayload,
    MultiKeyUnionPayload,
    ResolutionPlan,
    ResolutionStrategy,
    ResolutionVerdict,
    derive_resolution_verdict,
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
from sema.models.planner.target_model import (
    FieldPresence,
    RowPredicate,
    TargetObligation,
)


def _run_prov(**overrides: object) -> RunProvenance:
    base: dict[str, object] = dict(
        run_id="run-1",
        target_model_version="omop-cdm-5.4",
        target_schema_snapshot_hash="t",
        vocab_release="v1",
        context_card_version="cards-v3",
        prompt_template_version="tpl-7",
        few_shot_set_version="fs-12",
        constraint_version="rules-v2",
        llm_model="claude-opus-4.7",
        embedding_model="bge-large",
    )
    base.update(overrides)
    return RunProvenance(**base)


def _src(source_id: str = "cbioportal_gbm") -> SourceScope:
    return SourceScope(
        source_id=source_id,
        source_schema_hash=f"s-{source_id}",
        source_profile_hash=f"p-{source_id}",
    )


def _provenance(source_id: str = "cbioportal_gbm") -> Provenance:
    return Provenance(
        run=_run_prov(),
        source=_src(source_id),
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def migrated_neo4j(clean_neo4j):
    with clean_neo4j.session() as session:
        for stmt in cypher_up(enterprise=False, apoc=False):
            session.run(stmt)
    yield clean_neo4j
    with clean_neo4j.session() as session:
        for stmt in cypher_down(apoc=False):
            session.run(stmt)


def test_migration_creates_constraints(migrated_neo4j) -> None:
    with migrated_neo4j.session() as s:
        rows = list(s.run("SHOW CONSTRAINTS"))
    names = {r["name"] for r in rows}
    assert "MappingAssertion_id_unique" in names
    assert "HumanPin_id_unique" in names


def test_mapping_assertion_round_trip(migrated_neo4j) -> None:
    payload = VocabLookup(
        vocabulary_ref="omop.SNOMED",
        source_value_ref="cbio.cancer_type",
        domain_constraint_ref="omop.domain.Condition",
        require_standard=True,
        allow_zero_default=False,
        resolver_policy_ref="omop.snomed.condition.v1",
        effective_date_ref="cbio.diagnosis_date",
    )
    risk = RiskFlag(
        code=RiskCode.RISK_VOCAB_DOMAIN_MISMATCH,
        severity=Severity.warn,
        evidence=[
            Evidence(
                mode=EvidenceMode.CATEGORICAL,
                payload={"shape": "alpha"},
                sensitivity_class=SensitivityClass.PHI,
                source_ref="cbio.cancer_type",
            ),
            Evidence(
                mode=EvidenceMode.COUNT_ONLY,
                payload={"count": 7},
                sensitivity_class=SensitivityClass.PHI,
                source_ref="cbio.cancer_type",
            ),
        ],
        source_stage=SourceStage.producer,
        suggested_action=SuggestedAction.review,
    )
    assertion = MappingAssertion(
        id="a-rt",
        source_field_ref="cbio.cancer_type",
        target_property_ref="omop.condition_occurrence.condition_concept_id",
        pattern=MappingPattern.VOCAB_LOOKUP,
        payload=payload,
        confidence=0.91,
        risk_flags=[risk],
        provenance=_provenance(),
        status=Status.candidate,
    )
    with migrated_neo4j.session() as s:
        write_mapping_assertion(s, assertion)
        rt = read_mapping_assertion(s, assertion.id)
    assert rt == assertion


def test_human_pin_round_trip_each_state(migrated_neo4j) -> None:
    for state in PinState:
        pin = HumanPin(
            pin_id=f"pin-{state.value}",
            assertion_id="a-1",
            pinned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            pinned_by="reviewer@x",
            confirmed_under_run=_run_prov(),
            confirmed_under_source=_src(),
            pin_state=state,
        )
        with migrated_neo4j.session() as s:
            write_human_pin(s, pin)
            rt = read_human_pin(s, pin.pin_id)
        assert rt == pin


def test_resolution_plan_round_trip_multi_source(migrated_neo4j) -> None:
    plan = ResolutionPlan(
        id="r-mu",
        sources=[_src("acris.deeds"), _src("dof.parcels")],
        target_identity_ref="canonical.property_id",
        strategy=ResolutionStrategy.MULTI_KEY_UNION,
        payload=MultiKeyUnionPayload(
            source_key_refs=["acris.bbl", "dof.parcel_id"]
        ),
        confidence=0.85,
        provenance_run=_run_prov(),
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    with migrated_neo4j.session() as s:
        write_resolution_plan(s, plan)
        rt = read_resolution_plan(s, plan.id)
    assert rt == plan
    assert {s.source_id for s in rt.sources} == {"acris.deeds", "dof.parcels"}


def test_conflict_loser_relationship(migrated_neo4j) -> None:
    a_winner = MappingAssertion(
        id="a-win",
        source_field_ref="cbio.x",
        target_property_ref="omop.y",
        pattern=MappingPattern.DIRECT_COPY,
        payload=DirectCopyPayload(source_field_ref="cbio.x"),
        confidence=0.99,
        provenance=_provenance(),
        status=Status.auto_accepted,
    )
    a_loser = a_winner.model_copy(update={"id": "a-lose", "confidence": 0.7})
    winner = select_winner(
        [a_winner, a_loser], ConflictResolutionPolicy.default()
    )
    assert winner.id == "a-win"
    plan = MappingPlan(
        id="plan-conflict",
        source_scope_ref="cbio",
        obligation=TargetObligation(
            target_entity="omop.person",
            required_fields=["omop.y"],
            primary_key=PrimaryKeyStrategy.NATURAL_KEY,
        ),
        row_identity=RowIdentity(
            target_row_key_rule="hash",
            source_lineage=["cbio.x"],
            materialization_mode=MaterializationMode.MERGE,
        ),
        field_maps=[
            FieldMap(
                target_field_ref="omop.y",
                pattern=MappingPattern.DIRECT_COPY,
                payload=DirectCopyPayload(source_field_ref="cbio.x"),
            )
        ],
        risk_flags=[
            RiskFlag(
                code=RiskCode.RISK_ASSEMBLER_CONFLICT_RESOLVED,
                severity=Severity.info,
                evidence=[
                    Evidence(
                        mode=EvidenceMode.COUNT_ONLY,
                        payload={"count": 1},
                        sensitivity_class=SensitivityClass.PUBLIC,
                        source_ref="a-lose",
                    )
                ],
                source_stage=SourceStage.constraint,
                suggested_action=SuggestedAction.ignore_with_reason,
            )
        ],
        lineage=["cbio.x"],
    )
    assert plan.derive_verdict() == PlanVerdict.compilable
    with migrated_neo4j.session() as s:
        write_mapping_plan(s, plan)
        s.run(
            "CREATE (w:MappingAssertion {id: $win})\n"
            "CREATE (l:MappingAssertion {id: $lose})\n"
            "WITH w, l\n"
            "MATCH (p:MappingPlan {id: $plan_id})\n"
            "MERGE (w)-[:ASSEMBLED_INTO]->(p)\n"
            "MERGE (p)-[:CONFLICT_LOSER]->(l)",
            plan_id=plan.id,
            win=a_winner.id,
            lose=a_loser.id,
        )
        rt_plan = read_mapping_plan(s, plan.id)
        loser_row = s.run(
            "MATCH (p:MappingPlan {id: $id})-[:CONFLICT_LOSER]->(l) "
            "RETURN l.id AS loser",
            id=plan.id,
        ).single()
    assert rt_plan == plan
    assert loser_row["loser"] == "a-lose"


def test_pin_staleness_query_uses_pin_state_index(migrated_neo4j) -> None:
    pin = HumanPin(
        pin_id="pin-q",
        assertion_id="a-1",
        pinned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        pinned_by="reviewer@x",
        confirmed_under_run=_run_prov(),
        confirmed_under_source=_src(),
        pin_state=PinState.stale,
    )
    props = confirmed_under_to_properties(
        pin.confirmed_under_run, pin.confirmed_under_source
    )
    props["id"] = pin.pin_id
    props["pin_id"] = pin.pin_id
    props["assertion_id"] = pin.assertion_id
    props["pinned_at"] = pin.pinned_at.isoformat()
    props["pinned_by"] = pin.pinned_by
    props["pin_state"] = pin.pin_state.value
    props["expires_on_change_of"] = list(pin.expires_on_change_of)
    with migrated_neo4j.session() as s:
        s.run("CREATE (n:HumanPin) SET n = $props", props=props)
        plan_root = s.run(
            "EXPLAIN MATCH (h:HumanPin) WHERE h.pin_state IN $states "
            "RETURN h.pin_id",
            states=[PinState.active.value, PinState.stale.value],
        ).consume().plan
    operators = _collect_plan_operators(plan_root)
    assert any(
        "NodeIndexSeek" in op or "IndexSeek" in op for op in operators
    ), f"expected NodeIndexSeek operator on human_pin_state, got {operators}"


def _collect_plan_operators(plan: dict) -> list[str]:
    operators = [plan.get("operatorType", "")]
    for child in plan.get("children", []) or []:
        operators.extend(_collect_plan_operators(child))
    return [op for op in operators if op]


def test_cache_key_changes_across_runs() -> None:
    art = PromptArtifact.build(
        prefix_text="prefix",
        suffix_text="s",
        versions={"context_card_version": "v1"},
    )
    rp1 = _run_prov(context_card_version="cards-v3")
    rp2 = _run_prov(context_card_version="cards-v4")
    assert derive_cache_key(art, rp1) != derive_cache_key(art, rp2)


def test_resolution_verdict_derivation_matrix() -> None:
    assert (
        derive_resolution_verdict(
            produced_for_every_input=True,
            ambiguous_assignments=False,
            cycle_blocked=False,
            any_block_flag=False,
            plan_review_pending=False,
        )
        == ResolutionVerdict.resolved
    )
    assert (
        derive_resolution_verdict(
            produced_for_every_input=True,
            ambiguous_assignments=True,
            cycle_blocked=False,
            any_block_flag=False,
            plan_review_pending=False,
        )
        == ResolutionVerdict.ambiguous
    )


def test_multi_source_assertions_share_run_id(migrated_neo4j) -> None:
    a_cbio = MappingAssertion(
        id="a-cbio",
        source_field_ref="cbio.gender",
        target_property_ref="omop.gender_concept_id",
        pattern=MappingPattern.DIRECT_COPY,
        payload=DirectCopyPayload(source_field_ref="cbio.gender"),
        confidence=0.9,
        provenance=Provenance(
            run=_run_prov(),
            source=_src("cbioportal_gbm"),
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        status=Status.candidate,
    )
    a_msk = a_cbio.model_copy(
        update={
            "id": "a-msk",
            "provenance": Provenance(
                run=_run_prov(),
                source=_src("msk_chord"),
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        }
    )
    for a in (a_cbio, a_msk):
        with migrated_neo4j.session() as s:
            write_mapping_assertion(s, a)
    with migrated_neo4j.session() as s:
        rows = list(
            s.run(
                "MATCH (n:MappingAssertion) "
                "RETURN n.prov_run_run_id AS run, n.prov_source_source_id AS src"
            )
        )
    runs = {r["run"] for r in rows}
    sources = {r["src"] for r in rows}
    assert runs == {"run-1"}
    assert sources == {"cbioportal_gbm", "msk_chord"}
