import json
import pytest
from unittest.mock import MagicMock, call
from datetime import datetime, timezone

pytestmark = pytest.mark.unit

from sema.graph.loader import GraphLoader
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


@pytest.fixture
def loader(mock_driver):
    driver, _ = mock_driver
    return GraphLoader(driver)


def _make_assertion(subject, predicate, payload=None, object_ref=None, source="test",
                    confidence=0.9, status=AssertionStatus.AUTO, run_id="run-1"):
    return Assertion(
        id=f"a-{subject}-{predicate.value}",
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


class TestUpsertPhysicalNodes:
    def test_upsert_catalog(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_catalog("cdm")
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":Catalog" in cypher

    def test_upsert_schema(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_schema("clinical", "cdm")
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":Schema" in cypher
        assert "IN_CATALOG" in cypher

    def test_upsert_table(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_table("cancer_diagnosis", "clinical", "cdm", table_type="TABLE", comment="Diagnosis records")
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":Table" in cypher
        assert "IN_SCHEMA" in cypher

    def test_upsert_column(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_column("dx_type_cd", "cancer_diagnosis", "clinical", "cdm",
                            data_type="STRING", nullable=True, comment="Type code")
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":Column" in cypher
        assert "IN_TABLE" in cypher


class TestUpsertSemanticNodes:
    def test_upsert_entity(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_entity("Cancer Diagnosis", description="Primary dx",
                            source="llm", confidence=0.8, table_name="cancer_diagnosis",
                            schema_name="clinical", catalog="cdm")
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":Entity" in cypher
        assert "IMPLEMENTED_BY" in cypher

    def test_upsert_property(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_property("Diagnosis Type", semantic_type="categorical",
                              source="llm", confidence=0.8,
                              entity_name="Cancer Diagnosis",
                              column_name="dx_type_cd", table_name="cancer_diagnosis",
                              schema_name="clinical", catalog="cdm")
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":Property" in cypher

    def test_upsert_term(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_term("CRC", "Colorectal Cancer", source="llm", confidence=0.85)
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":Term" in cypher

    def test_upsert_value_set(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_value_set("oncotree_types", column_name="dx_type_cd",
                               table_name="cancer_diagnosis",
                               schema_name="clinical", catalog="cdm")
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":ValueSet" in cypher
        assert "STORED_IN" in cypher

    def test_upsert_synonym(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_synonym("colon cancer", parent_label=":Entity", parent_name="Cancer Diagnosis",
                             source="llm", confidence=0.8)
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":Synonym" in cypher
        assert "SYNONYM_OF" in cypher


class TestAssertionSupersession:
    def test_store_assertion_creates_node(self, loader, mock_driver):
        _, session = mock_driver
        assertion = _make_assertion(
            "unity://cdm.clinical.tbl.col",
            AssertionPredicate.HAS_LABEL,
            {"value": "My Label"},
        )
        loader.store_assertion(assertion)
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "CREATE" in cypher or "MERGE" in cypher
        assert ":Assertion" in cypher

    def test_supersession_marks_old_as_superseded(self, loader, mock_driver):
        _, session = mock_driver
        assertion = _make_assertion(
            "unity://cdm.clinical.tbl.col",
            AssertionPredicate.HAS_LABEL,
            {"value": "New Label"},
            run_id="run-2",
        )
        loader.store_assertion(assertion)
        calls = [str(c) for c in session.run.call_args_list]
        # Should have a call that sets status to 'superseded' for old assertions
        assert any("superseded" in c for c in calls)

    def test_pinned_assertions_not_superseded(self, loader, mock_driver):
        _, session = mock_driver
        assertion = _make_assertion(
            "unity://cdm.clinical.tbl.col",
            AssertionPredicate.HAS_LABEL,
            {"value": "New Label"},
            run_id="run-2",
        )
        loader.store_assertion(assertion)
        calls = [str(c) for c in session.run.call_args_list]
        # Supersession query should exclude pinned/accepted/rejected
        supersede_calls = [c for c in calls if "superseded" in c]
        assert all("auto" in c for c in supersede_calls)


class TestCandidateJoins:
    def test_upsert_candidate_join(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_candidate_join(
            from_table="cancer_diagnosis", from_schema="clinical", from_catalog="cdm",
            to_table="cancer_surgery", to_schema="clinical", to_catalog="cdm",
            on_column="patient_id", cardinality="one-to-many",
            source="heuristic", confidence=0.8,
        )
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "CANDIDATE_JOIN" in cypher


class TestBatchOperations:
    def test_batch_upsert_columns(self, loader, mock_driver):
        _, session = mock_driver
        columns = [
            {"name": "col1", "data_type": "STRING", "nullable": True, "comment": None},
            {"name": "col2", "data_type": "INT", "nullable": False, "comment": "test"},
        ]
        loader.batch_upsert_columns(columns, "tbl", "schema", "catalog")
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "UNWIND" in cypher
