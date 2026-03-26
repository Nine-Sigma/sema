import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

pytestmark = pytest.mark.unit

from sema.engine.resolution import ResolutionEngine
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)
from sema.graph.loader import GraphLoader


def _a(subject, predicate, payload=None, source="unity_catalog", confidence=0.9,
       status=AssertionStatus.AUTO, run_id="run-1", object_ref=None):
    return Assertion(
        id=f"a-{hash(subject + predicate.value + source) % 100000}",
        subject_ref=subject,
        predicate=predicate,
        payload=payload or {},
        object_ref=object_ref,
        source=source,
        confidence=confidence,
        status=status,
        run_id=run_id,
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def mock_loader():
    return MagicMock(spec=GraphLoader)


@pytest.fixture
def engine(mock_loader):
    return ResolutionEngine(mock_loader)


class TestSingleSourceResolution:
    def test_entity_created_from_assertion(self, engine, mock_loader):
        assertions = [
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "Cancer Diagnosis", "description": "Primary dx"},
               source="llm_interpretation", confidence=0.75),
        ]
        engine.resolve(assertions)
        mock_loader.upsert_entity.assert_called_once()
        call_kwargs = mock_loader.upsert_entity.call_args[1]
        assert call_kwargs["name"] == "Cancer Diagnosis"
        assert call_kwargs["source"] == "llm_interpretation"
        assert call_kwargs["confidence"] == 0.75

    def test_property_created_from_assertions(self, engine, mock_loader):
        assertions = [
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "Cancer Diagnosis"}, source="llm_interpretation"),
            _a("unity://cdm.clinical.tbl.col", AssertionPredicate.HAS_PROPERTY_NAME,
               {"value": "Diagnosis Type"}, source="llm_interpretation"),
            _a("unity://cdm.clinical.tbl.col", AssertionPredicate.HAS_SEMANTIC_TYPE,
               {"value": "categorical"}, source="llm_interpretation"),
        ]
        engine.resolve(assertions)
        mock_loader.upsert_property.assert_called_once()

    def test_term_and_valueset_from_decoded_values(self, engine, mock_loader):
        assertions = [
            _a("unity://cdm.clinical.tbl.col", AssertionPredicate.HAS_DECODED_VALUE,
               {"raw": "CRC", "label": "Colorectal Cancer"}, source="llm_interpretation"),
            _a("unity://cdm.clinical.tbl.col", AssertionPredicate.HAS_DECODED_VALUE,
               {"raw": "BRCA", "label": "Breast Cancer"}, source="llm_interpretation"),
        ]
        engine.resolve(assertions)
        assert mock_loader.upsert_term.call_count == 2
        assert mock_loader.upsert_value_set.call_count == 1
        assert mock_loader.add_term_to_value_set.call_count == 2

    def test_provenance_on_canonical_nodes(self, engine, mock_loader):
        assertions = [
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "Test Entity"}, source="llm_interpretation", confidence=0.8),
        ]
        engine.resolve(assertions)
        call_kwargs = mock_loader.upsert_entity.call_args[1]
        assert call_kwargs["source"] == "llm_interpretation"
        assert call_kwargs["confidence"] == 0.8


class TestMultiSourcePrecedence:
    def test_atlan_wins_for_business_definitions(self, engine, mock_loader):
        assertions = [
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "LLM Guess"}, source="llm_interpretation", confidence=0.7),
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "Atlan Curated Name"}, source="atlan", confidence=1.0),
        ]
        engine.resolve(assertions)
        call_kwargs = mock_loader.upsert_entity.call_args[1]
        assert call_kwargs["name"] == "Atlan Curated Name"

    def test_dbt_wins_over_llm_for_business(self, engine, mock_loader):
        assertions = [
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "LLM Guess"}, source="llm_interpretation", confidence=0.7),
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "dbt Description"}, source="dbt", confidence=0.9),
        ]
        engine.resolve(assertions)
        call_kwargs = mock_loader.upsert_entity.call_args[1]
        assert call_kwargs["name"] == "dbt Description"


class TestHumanOverrides:
    def test_pinned_always_wins(self, engine, mock_loader):
        assertions = [
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "Atlan Name"}, source="atlan", confidence=1.0),
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "Human Override"}, source="human", confidence=1.0,
               status=AssertionStatus.PINNED),
        ]
        engine.resolve(assertions)
        call_kwargs = mock_loader.upsert_entity.call_args[1]
        assert call_kwargs["name"] == "Human Override"

    def test_rejected_excluded(self, engine, mock_loader):
        assertions = [
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "Bad Name"}, source="llm_interpretation", confidence=0.9,
               status=AssertionStatus.REJECTED),
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "Good Name"}, source="llm_interpretation", confidence=0.7),
        ]
        engine.resolve(assertions)
        call_kwargs = mock_loader.upsert_entity.call_args[1]
        assert call_kwargs["name"] == "Good Name"

    def test_superseded_excluded(self, engine, mock_loader):
        assertions = [
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "Old Name"}, source="llm_interpretation",
               status=AssertionStatus.SUPERSEDED),
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "New Name"}, source="llm_interpretation", confidence=0.8),
        ]
        engine.resolve(assertions)
        call_kwargs = mock_loader.upsert_entity.call_args[1]
        assert call_kwargs["name"] == "New Name"


class TestIdempotency:
    def test_resolve_twice_same_result(self, engine, mock_loader):
        assertions = [
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "Entity"}, source="llm_interpretation"),
        ]
        engine.resolve(assertions)
        first_call = mock_loader.upsert_entity.call_args

        mock_loader.reset_mock()
        engine.resolve(assertions)
        second_call = mock_loader.upsert_entity.call_args

        assert first_call == second_call


class TestSynonymResolution:
    def test_synonym_creates_alias_node(self, engine, mock_loader):
        assertions = [
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "Cancer Diagnosis"}, source="llm_interpretation"),
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ALIAS,
               {"value": "cancer dx", "is_preferred": True}, source="llm_interpretation"),
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ALIAS,
               {"value": "diagnosis", "is_preferred": False}, source="llm_interpretation"),
        ]
        engine.resolve(assertions)
        assert mock_loader.upsert_alias.call_count == 2

    def test_hierarchy_creates_parent_of_edges(self, engine, mock_loader):
        assertions = [
            _a("unity://cdm.clinical.tbl.col", AssertionPredicate.PARENT_OF,
               {"parent": "Stage III", "child": "Stage IIIA"}, source="pattern_match"),
            _a("unity://cdm.clinical.tbl.col", AssertionPredicate.PARENT_OF,
               {"parent": "Stage III", "child": "Stage IIIB"}, source="pattern_match"),
        ]
        engine.resolve(assertions)
        assert mock_loader.add_term_hierarchy.call_count == 2


class TestAssertionStorage:
    def test_all_assertions_stored(self, engine, mock_loader):
        assertions = [
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "Entity"}, source="llm_interpretation"),
            _a("unity://cdm.clinical.tbl", AssertionPredicate.HAS_ALIAS,
               {"value": "syn", "is_preferred": True}, source="llm_interpretation"),
        ]
        engine.resolve(assertions)
        assert mock_loader.store_assertion.call_count == 2
