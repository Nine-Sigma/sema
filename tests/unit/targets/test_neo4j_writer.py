"""Unit checks for Neo4jGraphWriter using a mock driver (5.2)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from sema.models.planner._enums import ModelRole
from sema.targets.materializer_ops import (
    ConstraintOp,
    ContextCardOp,
    CurrentFlipOp,
    EnrichmentDecisionOp,
    EntityOp,
    PropertyOp,
    RelationshipOp,
    TargetObligationOp,
    TermOp,
    VocabularyBindingOp,
)
from sema.targets.neo4j_writer import Neo4jGraphWriter

pytestmark = pytest.mark.unit


def _mock_driver_capturing(calls: list[tuple[str, dict]]) -> MagicMock:
    driver = MagicMock()
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.side_effect = lambda cypher, **kwargs: calls.append(
        (cypher, kwargs)
    )
    driver.session.return_value = session
    return driver


def _entity_op(qname: str = "fake.person") -> EntityOp:
    return EntityOp(
        target_model_id="fake-target",
        target_model_version="1.0.0",
        target_schema_snapshot_hash="h" * 64,
        qualified_name=qname,
        kind="TABLE_ROW",
        enrichment_status={
            "structure": "not_required",
            "obligations": "not_required",
            "vocabulary_bindings": "required_deferred",
            "semantic_aliases": "required_deferred",
            "terms": "not_required",
        },
    )


def test_write_entity_emits_merge_with_target_role() -> None:
    calls: list[tuple[str, dict]] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    writer.write_entity(_entity_op())
    assert len(calls) == 1
    cypher, params = calls[0]
    assert "MERGE (n:Entity" in cypher
    assert "qualified_name: $qualified_name" in cypher
    assert params["model_role"] == ModelRole.TARGET.value
    assert params["is_current"] is True
    assert params["enrichment_vocabulary_bindings_status"] == "required_deferred"


def test_write_property_emits_full_hash_versioned_merge_keys() -> None:
    calls: list[tuple[str, dict]] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    op = PropertyOp(
        target_model_id="t",
        target_model_version="1",
        target_schema_snapshot_hash="h",
        parent_entity_qualified_name="e",
        name="p",
        type="string",
        nullable=False,
    )
    writer.write_property(op)
    cypher, params = calls[0]
    assert "MERGE (n:Property" in cypher
    assert "parent_entity_qualified_name: $parent_entity_qualified_name" in cypher
    assert params["property_kind"] == "COLUMN"
    assert params["materialized_as_edge_property"] is True


def test_write_vocabulary_binding_carries_standard_domain_governed() -> None:
    calls: list[tuple[str, dict]] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    op = VocabularyBindingOp(
        target_model_id="t",
        target_model_version="1",
        target_schema_snapshot_hash="h",
        parent_entity_qualified_name="omop.condition_occurrence",
        property_name="condition_concept_id",
        vocabulary_name="SNOMED",
        vocabulary_source="EXTERNAL",
        domain="Condition",
        require_standard=True,
        standard_domain_governed=True,
    )
    writer.write_vocabulary_binding(op)
    cypher, params = calls[0]
    assert "n.standard_domain_governed = $standard_domain_governed" in cypher
    assert params["standard_domain_governed"] is True


def test_write_endpoint_property_carries_endpoint_typing() -> None:
    calls: list[tuple[str, dict]] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    op = PropertyOp(
        target_model_id="t",
        target_model_version="1",
        target_schema_snapshot_hash="h",
        parent_entity_qualified_name="acris.OWNS",
        name="subject",
        type="ref",
        nullable=False,
        property_kind="ENDPOINT",
        endpoint_role="subject",
        endpoint_target_entity_qualified_name="acris.LLC",
        endpoint_cardinality="one",
        endpoint_nullable=False,
        materialized_as_edge_property=False,
    )
    writer.write_property(op)
    _, params = calls[0]
    assert params["property_kind"] == "ENDPOINT"
    assert params["endpoint_role"] == "subject"
    assert params["endpoint_target_entity_qualified_name"] == "acris.LLC"
    assert params["endpoint_cardinality"] == "one"
    assert params["endpoint_nullable"] is False
    assert params["materialized_as_edge_property"] is False


def test_write_constraint_uses_payload_hash_in_merge_key() -> None:
    calls: list[tuple[str, dict]] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    op = ConstraintOp(
        target_model_id="t",
        target_model_version="1",
        target_schema_snapshot_hash="h",
        attached_property_id="e.p",
        constraint_kind="domain_binding",
        payload={"domain_id": "x"},
        payload_hash="abc123",
    )
    writer.write_constraint(op)
    cypher, params = calls[0]
    assert "MERGE (n:Constraint" in cypher
    assert "payload_hash: $payload_hash" in cypher
    assert params["payload_hash"] == "abc123"
    assert params["model_role"] == ModelRole.TARGET.value


def test_write_target_obligation_serialises_payload_json() -> None:
    calls: list[tuple[str, dict]] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    op = TargetObligationOp(
        target_model_id="t",
        target_model_version="1",
        target_schema_snapshot_hash="h",
        target_entity="fake.person",
        payload={"required_fields": ["person_id"], "primary_key": "NATURAL_KEY"},
    )
    writer.write_target_obligation(op)
    cypher, params = calls[0]
    assert "MERGE (n:TargetObligation" in cypher
    assert "person_id" in params["payload_json"]
    assert params["model_role"] == ModelRole.TARGET.value


def test_write_enrichment_decision_emits_decisions_json() -> None:
    calls: list[tuple[str, dict]] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    op = EnrichmentDecisionOp(
        target_model_id="t",
        target_model_version="1",
        target_schema_snapshot_hash="h",
        entity_ref="fake.person",
        decisions_json='{"structure":{"status":"not_required"}}',
        decided_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    writer.write_enrichment_decision(op)
    cypher, params = calls[0]
    assert "MERGE (n:EnrichmentDecision" in cypher
    assert params["decisions_json"].startswith("{")
    assert "2026-01-01" in params["decided_at"]


def test_write_relationship_matches_both_endpoints_by_full_keys() -> None:
    calls: list[tuple[str, dict]] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    op = RelationshipOp(
        rel_type="HAS_PROPERTY",
        target_schema_snapshot_hash="h",
        from_label="Entity",
        from_keys={
            "target_model_id": "t",
            "target_model_version": "1",
            "target_schema_snapshot_hash": "h",
            "qualified_name": "e",
        },
        to_label="Property",
        to_keys={
            "target_model_id": "t",
            "target_model_version": "1",
            "target_schema_snapshot_hash": "h",
            "parent_entity_qualified_name": "e",
            "name": "p",
        },
    )
    writer.write_relationship(op)
    cypher, params = calls[0]
    assert "MATCH (a:Entity)" in cypher
    assert "MATCH (b:Property)" in cypher
    assert "HAS_PROPERTY" in cypher
    assert params["from_qualified_name"] == "e"
    assert params["to_name"] == "p"


def test_flip_prior_generations_filters_by_logical_identity() -> None:
    calls: list[tuple[str, dict]] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    op = CurrentFlipOp(
        target_model_id="t",
        target_model_version="1",
        current_snapshot_hash="h-current",
        entity_qualified_names=("fake.person",),
        property_keys=(("fake.person", "person_id"),),
        obligation_target_entities=("fake.person",),
        enrichment_entity_refs=("fake.person",),
    )
    writer.flip_prior_generations(op)
    labels = [c[0] for c in calls]
    assert any(":Entity)" in c for c in labels)
    assert any(":Property)" in c for c in labels)
    assert any(":TargetObligation)" in c for c in labels)
    assert any(":EnrichmentDecision)" in c for c in labels)
    assert any(":Constraint)" in c for c in labels)
    for cypher, params in calls:
        assert "n.target_schema_snapshot_hash <> $current_hash" in cypher
        assert "SET n.is_current = false" in cypher
        assert params["current_hash"] == "h-current"


def test_flip_skips_empty_buckets() -> None:
    calls: list[tuple[str, dict]] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    op = CurrentFlipOp(
        target_model_id="t",
        target_model_version="1",
        current_snapshot_hash="h-current",
        entity_qualified_names=("only.entity",),
    )
    writer.flip_prior_generations(op)
    assert len(calls) == 1


def test_target_obligation_merge_sets_is_current_true() -> None:
    calls: list[tuple[str, dict]] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    op = TargetObligationOp(
        target_model_id="t",
        target_model_version="1",
        target_schema_snapshot_hash="h",
        target_entity="fake.person",
        payload={"required_fields": ["person_id"]},
    )
    writer.write_target_obligation(op)
    cypher, params = calls[0]
    assert "n.is_current = $is_current" in cypher
    assert params["is_current"] is True


def test_term_merge_sets_is_current_true() -> None:
    calls: list[tuple[str, dict]] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    op = TermOp(
        target_model_id="t",
        target_model_version="1",
        target_schema_snapshot_hash="h",
        vocabulary_name="GENDER_CV",
        code="M",
        display="Male",
    )
    writer.write_term(op)
    cypher, params = calls[0]
    assert "n.is_current = $is_current" in cypher
    assert params["is_current"] is True


def test_vocabulary_binding_merge_keyed_on_full_tuple() -> None:
    calls: list[tuple[str, dict]] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    op = VocabularyBindingOp(
        target_model_id="t",
        target_model_version="1",
        target_schema_snapshot_hash="h",
        parent_entity_qualified_name="omop.person",
        property_name="gender_concept_id",
        vocabulary_name="SNOMED",
        vocabulary_source="EXTERNAL",
        domain="Gender",
        require_standard=True,
        allow_zero_default=False,
        effective_date_ref="visit.start_date",
        resolver_policy_ref="omop.snomed.gender.v1",
    )
    writer.write_vocabulary_binding(op)
    cypher, params = calls[0]
    assert "MERGE (n:VocabularyBinding" in cypher
    assert "vocabulary_name: $vocabulary_name" in cypher
    assert params["domain"] == "Gender"
    assert params["require_standard"] is True
    assert params["effective_date_ref"] == "visit.start_date"
    assert params["resolver_policy_ref"] == "omop.snomed.gender.v1"
    assert params["model_role"] == ModelRole.TARGET.value


def test_context_card_merge_carries_card_hash_and_content() -> None:
    calls: list[tuple[str, dict]] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    op = ContextCardOp(
        target_model_id="t",
        target_model_version="1",
        target_schema_snapshot_hash="h",
        entity_qualified_name="omop.person",
        card_version="1.0.0",
        card_hash="0" * 64,
        description="OMOP person row.",
        examples=["sample"],
        obligation_summary="PK + gender concept",
        curated_synonyms=["patient"],
    )
    writer.write_context_card(op)
    cypher, params = calls[0]
    assert "MERGE (n:ContextCard" in cypher
    assert "n.card_hash = $card_hash" in cypher
    assert params["card_version"] == "1.0.0"
    assert params["card_hash"] == "0" * 64
    assert params["examples"] == ["sample"]
    assert params["curated_synonyms"] == ["patient"]
    assert params["model_role"] == ModelRole.TARGET.value


def test_flip_covers_obligations_terms_constraints_bindings_cards() -> None:
    calls: list[tuple[str, dict]] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    op = CurrentFlipOp(
        target_model_id="t",
        target_model_version="1",
        current_snapshot_hash="h-current",
        entity_qualified_names=("e",),
        property_keys=(("e", "p"),),
        obligation_target_entities=("e",),
        enrichment_entity_refs=("e",),
        vocabulary_binding_keys=(("e", "p", "SNOMED"),),
        context_card_keys=(("e", "1.0.0"),),
        term_keys=(("SNOMED", "12345"),),
    )
    writer.flip_prior_generations(op)
    bodies = [c[0] for c in calls]
    assert any(":TargetObligation" in b for b in bodies)
    assert any(":Term" in b for b in bodies)
    assert any(":Constraint" in b for b in bodies)
    assert any(":VocabularyBinding" in b for b in bodies)
    assert any(":ContextCard" in b for b in bodies)
    for cypher, params in calls:
        assert "$current_hash" in cypher
        assert params["current_hash"] == "h-current"


def test_term_merge_keyed_on_full_hash_versioned_tuple() -> None:
    calls: list[tuple[str, dict]] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    op = TermOp(
        target_model_id="t",
        target_model_version="1",
        target_schema_snapshot_hash="h",
        vocabulary_name="GENDER_CV",
        code="M",
        display="Male",
    )
    writer.write_term(op)
    cypher, params = calls[0]
    assert "vocabulary_name: $vocabulary_name" in cypher
    assert "code: $code" in cypher
    assert params["model_role"] == ModelRole.TARGET.value
