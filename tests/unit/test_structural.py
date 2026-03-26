import pytest
from unittest.mock import MagicMock, call
from datetime import datetime, timezone

pytestmark = pytest.mark.unit

from sema.engine.structural import StructuralEngine
from sema.models.assertions import Assertion, AssertionPredicate, AssertionStatus
from sema.graph.loader import GraphLoader


def _make_assertion(subject, predicate, payload=None, object_ref=None, run_id="run-1"):
    return Assertion(
        id=f"a-{hash(subject + predicate.value) % 10000}",
        subject_ref=subject,
        predicate=predicate,
        payload=payload or {},
        object_ref=object_ref,
        source="unity_catalog",
        confidence=1.0,
        run_id=run_id,
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def mock_loader():
    return MagicMock(spec=GraphLoader)


@pytest.fixture
def engine(mock_loader):
    return StructuralEngine(mock_loader)


class TestCatalogSchemaTable:
    def test_creates_catalog_node(self, engine, mock_loader):
        assertions = [
            _make_assertion("unity://cdm.clinical.cancer_diagnosis",
                          AssertionPredicate.TABLE_EXISTS, {"table_type": "TABLE"}),
        ]
        engine.process(assertions)
        mock_loader.upsert_catalog.assert_called_with("cdm")

    def test_creates_schema_node(self, engine, mock_loader):
        assertions = [
            _make_assertion("unity://cdm.clinical.cancer_diagnosis",
                          AssertionPredicate.TABLE_EXISTS, {"table_type": "TABLE"}),
        ]
        engine.process(assertions)
        mock_loader.upsert_schema.assert_called_with("clinical", "cdm")

    def test_creates_table_node(self, engine, mock_loader):
        assertions = [
            _make_assertion("unity://cdm.clinical.cancer_diagnosis",
                          AssertionPredicate.TABLE_EXISTS, {"table_type": "TABLE"}),
        ]
        engine.process(assertions)
        mock_loader.upsert_table.assert_called_with(
            "cancer_diagnosis", "clinical", "cdm", table_type="TABLE", comment=None,
        )

    def test_table_with_comment(self, engine, mock_loader):
        assertions = [
            _make_assertion("unity://cdm.clinical.cancer_diagnosis",
                          AssertionPredicate.TABLE_EXISTS, {"table_type": "TABLE"}),
            _make_assertion("unity://cdm.clinical.cancer_diagnosis",
                          AssertionPredicate.HAS_COMMENT, {"value": "Diagnosis records"}),
        ]
        engine.process(assertions)
        mock_loader.upsert_table.assert_called_with(
            "cancer_diagnosis", "clinical", "cdm", table_type="TABLE", comment="Diagnosis records",
        )


class TestColumnNodes:
    def test_creates_column_node(self, engine, mock_loader):
        assertions = [
            _make_assertion("unity://cdm.clinical.tbl",
                          AssertionPredicate.TABLE_EXISTS, {"table_type": "TABLE"}),
            _make_assertion("unity://cdm.clinical.tbl.dx_type_cd",
                          AssertionPredicate.COLUMN_EXISTS,
                          {"data_type": "STRING", "nullable": True, "comment": "Type code"}),
        ]
        engine.process(assertions)
        mock_loader.upsert_column.assert_called_with(
            "dx_type_cd", "tbl", "clinical", "cdm",
            data_type="STRING", nullable=True, comment="Type code",
        )


class TestCandidateJoins:
    def test_fk_creates_join_path(self, engine, mock_loader):
        assertions = [
            _make_assertion("unity://cdm.clinical.cancer_diagnosis",
                          AssertionPredicate.TABLE_EXISTS, {"table_type": "TABLE"}),
            _make_assertion(
                "unity://cdm.clinical.cancer_diagnosis",
                AssertionPredicate.JOINS_TO,
                {"on_column": "patient_id", "to_column": "patient_id"},
                object_ref="unity://cdm.clinical.patients",
            ),
        ]
        engine.process(assertions)
        mock_loader.upsert_join_path.assert_called_once()
        call_kwargs = mock_loader.upsert_join_path.call_args[1]
        assert "patient_id" in call_kwargs["name"]


class TestMultipleTables:
    def test_processes_multiple_tables(self, engine, mock_loader):
        assertions = [
            _make_assertion("unity://cdm.clinical.tbl1",
                          AssertionPredicate.TABLE_EXISTS, {"table_type": "TABLE"}),
            _make_assertion("unity://cdm.clinical.tbl2",
                          AssertionPredicate.TABLE_EXISTS, {"table_type": "VIEW"}),
        ]
        engine.process(assertions)
        assert mock_loader.upsert_table.call_count == 2
