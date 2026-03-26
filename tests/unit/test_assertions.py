import pytest
from datetime import datetime, timezone

pytestmark = pytest.mark.unit

from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)


class TestAssertionModel:
    def test_create_unary_assertion(self):
        a = Assertion(
            id="test-1",
            subject_ref="unity://cdm.clinical.cancer_diagnosis.dx_type_cd",
            predicate=AssertionPredicate.HAS_LABEL,
            payload={"value": "Diagnosis Type Code"},
            source="unity_catalog",
            confidence=0.95,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
        )
        assert a.subject_ref == "unity://cdm.clinical.cancer_diagnosis.dx_type_cd"
        assert a.predicate == AssertionPredicate.HAS_LABEL
        assert a.payload["value"] == "Diagnosis Type Code"
        assert a.object_ref is None
        assert a.source == "unity_catalog"
        assert a.confidence == 0.95
        assert a.status == AssertionStatus.AUTO

    def test_create_relational_assertion(self):
        a = Assertion(
            id="test-2",
            subject_ref="unity://cdm.clinical.cancer_diagnosis",
            predicate=AssertionPredicate.JOINS_TO,
            object_ref="unity://cdm.clinical.cancer_surgery",
            payload={"on_column": "patient_id", "cardinality": "one-to-many"},
            source="heuristic",
            confidence=0.8,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
        )
        assert a.object_ref == "unity://cdm.clinical.cancer_surgery"
        assert a.payload["on_column"] == "patient_id"

    def test_assertion_status_default_is_auto(self):
        a = Assertion(
            id="test-3",
            subject_ref="test://col",
            predicate=AssertionPredicate.HAS_DATATYPE,
            payload={"value": "string"},
            source="test",
            confidence=0.9,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
        )
        assert a.status == AssertionStatus.AUTO

    def test_assertion_status_pinned(self):
        a = Assertion(
            id="test-4",
            subject_ref="test://col",
            predicate=AssertionPredicate.HAS_LABEL,
            payload={"value": "Override Name"},
            source="human",
            confidence=1.0,
            status=AssertionStatus.PINNED,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
        )
        assert a.status == AssertionStatus.PINNED

    def test_assertion_status_rejected(self):
        a = Assertion(
            id="test-5",
            subject_ref="test://col",
            predicate=AssertionPredicate.HAS_LABEL,
            payload={"value": "Bad Name"},
            source="llm_interpretation",
            confidence=0.6,
            status=AssertionStatus.REJECTED,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
        )
        assert a.status == AssertionStatus.REJECTED

    def test_assertion_status_superseded(self):
        a = Assertion(
            id="test-6",
            subject_ref="test://col",
            predicate=AssertionPredicate.HAS_LABEL,
            payload={"value": "Old Name"},
            source="llm_interpretation",
            confidence=0.6,
            status=AssertionStatus.SUPERSEDED,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
        )
        assert a.status == AssertionStatus.SUPERSEDED

    def test_assertion_json_serialization(self):
        a = Assertion(
            id="test-7",
            subject_ref="test://table",
            predicate=AssertionPredicate.TABLE_EXISTS,
            payload={"table_type": "TABLE"},
            source="unity_catalog",
            confidence=1.0,
            run_id="run-1",
            observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        data = a.model_dump(mode="json")
        assert data["subject_ref"] == "test://table"
        assert data["predicate"] == "table_exists"
        assert data["source"] == "unity_catalog"
        roundtrip = Assertion.model_validate(data)
        assert roundtrip.id == a.id

    def test_assertion_run_id_required(self):
        with pytest.raises(Exception):
            Assertion(
                id="test-8",
                subject_ref="test://col",
                predicate=AssertionPredicate.HAS_LABEL,
                payload={},
                source="test",
                confidence=0.5,
                observed_at=datetime.now(timezone.utc),
            )

    def test_assertion_confidence_bounds(self):
        a = Assertion(
            id="test-9",
            subject_ref="test://col",
            predicate=AssertionPredicate.HAS_LABEL,
            payload={},
            source="test",
            confidence=0.0,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
        )
        assert a.confidence == 0.0

        a2 = Assertion(
            id="test-10",
            subject_ref="test://col",
            predicate=AssertionPredicate.HAS_LABEL,
            payload={},
            source="test",
            confidence=1.0,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
        )
        assert a2.confidence == 1.0


class TestAssertionPredicates:
    def test_all_predicates_exist(self):
        expected = [
            "table_exists", "column_exists", "has_datatype", "has_label",
            "has_description", "has_comment", "has_tag", "has_top_values",
            "has_sample_rows", "joins_to", "has_entity_name",
            "has_property_name", "has_semantic_type", "has_decoded_value",
            "has_synonym", "vocabulary_match", "parent_of", "maps_to",
        ]
        predicate_values = [p.value for p in AssertionPredicate]
        for exp in expected:
            assert exp in predicate_values, f"Missing predicate: {exp}"


class TestAssertionStatus:
    def test_all_statuses_exist(self):
        expected = ["auto", "accepted", "rejected", "pinned", "superseded"]
        status_values = [s.value for s in AssertionStatus]
        for exp in expected:
            assert exp in status_values, f"Missing status: {exp}"
