import json
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

pytestmark = pytest.mark.unit

from sema.engine.joins import JoinInferenceEngine
from sema.models.assertions import Assertion, AssertionPredicate


def _make_col_assertion(table_ref, col_name, data_type="STRING", nullable=False):
    return Assertion(
        id=f"a-{hash(table_ref + col_name) % 100000}",
        subject_ref=f"{table_ref}.{col_name}",
        predicate=AssertionPredicate.COLUMN_EXISTS,
        payload={"data_type": data_type, "nullable": nullable, "comment": None},
        source="unity_catalog",
        confidence=1.0,
        run_id="run-1",
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


class TestColumnNameHeuristic:
    def test_shared_column_name_detected(self):
        engine = JoinInferenceEngine(run_id="test-run")
        table_columns = {
            "unity://cdm.clinical.cancer_diagnosis": ["patient_id", "dx_type_cd", "stage"],
            "unity://cdm.clinical.cancer_surgery": ["patient_id", "procedure_date"],
        }
        candidates = engine.find_heuristic_joins(table_columns)
        assert len(candidates) >= 1
        assert any(c["on_column"] == "patient_id" for c in candidates)

    def test_generic_id_column_lower_confidence(self):
        engine = JoinInferenceEngine(run_id="test-run")
        table_columns = {
            "unity://cdm.clinical.tbl1": ["id", "name"],
            "unity://cdm.clinical.tbl2": ["id", "value"],
        }
        candidates = engine.find_heuristic_joins(table_columns)
        id_candidates = [c for c in candidates if c["on_column"] == "id"]
        if id_candidates:
            assert id_candidates[0]["confidence"] < 0.7

    def test_no_shared_columns_no_candidates(self):
        engine = JoinInferenceEngine(run_id="test-run")
        table_columns = {
            "unity://cdm.clinical.tbl1": ["col_a", "col_b"],
            "unity://cdm.clinical.tbl2": ["col_c", "col_d"],
        }
        candidates = engine.find_heuristic_joins(table_columns)
        assert len(candidates) == 0


class TestJoinAssertions:
    def test_heuristic_join_emits_assertion(self):
        engine = JoinInferenceEngine(run_id="test-run")
        table_columns = {
            "unity://cdm.clinical.cancer_diagnosis": ["patient_id", "dx_type_cd"],
            "unity://cdm.clinical.cancer_surgery": ["patient_id", "procedure_date"],
        }
        assertions = engine.infer_joins(table_columns)
        join_assertions = [a for a in assertions if a.predicate == AssertionPredicate.JOINS_TO]
        assert len(join_assertions) >= 1
        assert join_assertions[0].source == "heuristic"
        assert join_assertions[0].payload["on_column"] == "patient_id"

    def test_join_has_both_subject_and_object(self):
        engine = JoinInferenceEngine(run_id="test-run")
        table_columns = {
            "unity://cdm.clinical.tbl1": ["patient_id"],
            "unity://cdm.clinical.tbl2": ["patient_id"],
        }
        assertions = engine.infer_joins(table_columns)
        for a in assertions:
            if a.predicate == AssertionPredicate.JOINS_TO:
                assert a.object_ref is not None
