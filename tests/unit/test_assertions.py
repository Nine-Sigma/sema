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
            subject_ref="databricks://ws/cdm/clinical/cancer_diagnosis/dx_type_cd",
            predicate=AssertionPredicate.HAS_LABEL,
            payload={"value": "Diagnosis Type Code"},
            source="unity_catalog",
            confidence=0.95,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
        )
        assert a.subject_ref == "databricks://ws/cdm/clinical/cancer_diagnosis/dx_type_cd"
        assert a.predicate == AssertionPredicate.HAS_LABEL
        assert a.payload["value"] == "Diagnosis Type Code"
        assert a.object_ref is None
        assert a.subject_id is None
        assert a.object_id is None
        assert a.status == AssertionStatus.AUTO

    def test_create_with_subject_id_and_object_id(self):
        a = Assertion(
            id="test-id",
            subject_ref="databricks://ws/cdm/clinical/diagnosis",
            predicate=AssertionPredicate.HAS_ENTITY_NAME,
            payload={"value": "Cancer Diagnosis"},
            source="llm_interpretation",
            confidence=0.85,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
            subject_id="ent-123",
            object_id="tbl-456",
        )
        assert a.subject_id == "ent-123"
        assert a.object_id == "tbl-456"

    def test_subject_id_object_id_default_none(self):
        a = Assertion(
            id="test-none",
            subject_ref="test://col",
            predicate=AssertionPredicate.HAS_LABEL,
            payload={},
            source="test",
            confidence=0.9,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
        )
        assert a.subject_id is None
        assert a.object_id is None

    def test_create_relational_assertion(self):
        a = Assertion(
            id="test-2",
            subject_ref="databricks://ws/cdm/clinical/cancer_diagnosis",
            predicate=AssertionPredicate.HAS_JOIN_EVIDENCE,
            object_ref="databricks://ws/cdm/clinical/cancer_surgery",
            payload={
                "join_predicates": [
                    {
                        "left_table": "databricks://ws/cdm/clinical/cancer_diagnosis",
                        "left_column": "patient_id",
                        "right_table": "databricks://ws/cdm/clinical/cancer_surgery",
                        "right_column": "patient_id",
                        "operator": "=",
                    }
                ],
                "hop_count": 1,
                "cardinality": "one-to-many",
            },
            source="heuristic",
            confidence=0.8,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
        )
        assert a.object_ref == "databricks://ws/cdm/clinical/cancer_surgery"
        assert a.payload["hop_count"] == 1

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
            subject_id="node-1",
        )
        data = a.model_dump(mode="json")
        assert data["subject_ref"] == "test://table"
        assert data["predicate"] == "table_exists"
        assert data["subject_id"] == "node-1"
        assert data["object_id"] is None
        roundtrip = Assertion.model_validate(data)
        assert roundtrip.id == a.id
        assert roundtrip.subject_id == "node-1"

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
            "has_sample_rows", "has_entity_name",
            "has_property_name", "has_semantic_type", "has_decoded_value",
            "vocabulary_match", "parent_of", "maps_to",
            # New predicates
            "has_alias", "has_join_evidence",
            "entity_on_table", "property_on_column",
        ]
        predicate_values = [p.value for p in AssertionPredicate]
        for exp in expected:
            assert exp in predicate_values, f"Missing predicate: {exp}"

    def test_deprecated_predicates_still_exist(self):
        assert AssertionPredicate.HAS_SYNONYM.value == "has_synonym"
        assert AssertionPredicate.JOINS_TO.value == "joins_to"

    def test_new_alias_predicate(self):
        assert AssertionPredicate.HAS_ALIAS.value == "has_alias"

    def test_new_join_evidence_predicate(self):
        assert AssertionPredicate.HAS_JOIN_EVIDENCE.value == "has_join_evidence"

    def test_entity_on_table_predicate(self):
        assert AssertionPredicate.ENTITY_ON_TABLE.value == "entity_on_table"

    def test_property_on_column_predicate(self):
        assert AssertionPredicate.PROPERTY_ON_COLUMN.value == "property_on_column"


class TestAssertionStatus:
    def test_all_statuses_exist(self):
        expected = ["auto", "accepted", "rejected", "pinned", "superseded"]
        status_values = [s.value for s in AssertionStatus]
        for exp in expected:
            assert exp in status_values, f"Missing status: {exp}"
