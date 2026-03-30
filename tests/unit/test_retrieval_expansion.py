"""Tests for type-aware retrieval expansion functions."""

import pytest
from unittest.mock import MagicMock, patch

from sema.pipeline.retrieval_utils import (
    _expand_alias_hit,
    _expand_property_hit,
    _expand_term_hit,
    merge_and_rank_candidates,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine._run_query.return_value = []
    return engine


class TestMergeAndRank:
    def test_scores_and_sorts(self) -> None:
        candidates = [
            {"score": 0.5, "confidence": 0.8, "match_type": "vector"},
            {"score": 0.9, "confidence": 0.9, "match_type": "vector"},
        ]
        ranked = merge_and_rank_candidates(candidates)
        assert ranked[0]["final_score"] > ranked[1]["final_score"]

    def test_lexical_boost(self) -> None:
        candidates = [
            {"score": 0.5, "confidence": 0.5, "match_type": "lexical_exact"},
            {"score": 0.5, "confidence": 0.5, "match_type": "vector"},
        ]
        ranked = merge_and_rank_candidates(candidates)
        assert ranked[0]["match_type"] == "lexical_exact"


class TestPropertyExpansion:
    def test_returns_property_candidate(self, mock_engine) -> None:
        hit = {
            "name": "diagnosis_code",
            "entity_name": "Diagnosis",
            "final_score": 0.85,
        }
        results = _expand_property_hit(mock_engine, hit)
        assert any(r["type"] == "property" for r in results)
        prop = next(r for r in results if r["type"] == "property")
        assert prop["name"] == "diagnosis_code"
        assert prop["entity_name"] == "Diagnosis"

    def test_expands_owning_entity(self, mock_engine) -> None:
        mock_engine._run_query.return_value = [
            {"table_name": "dx_table", "schema_name": "clinical",
             "catalog": "cdm", "columns": []},
        ]
        hit = {
            "name": "diagnosis_code",
            "entity_name": "Diagnosis",
            "final_score": 0.85,
        }
        results = _expand_property_hit(mock_engine, hit)
        entities = [r for r in results if r["type"] == "entity"]
        assert len(entities) == 1
        assert entities[0]["name"] == "Diagnosis"
        assert entities[0]["table"] == "dx_table"

    def test_no_entity_when_entity_name_missing(self, mock_engine) -> None:
        hit = {"name": "some_col", "entity_name": "", "final_score": 0.5}
        results = _expand_property_hit(mock_engine, hit)
        entities = [r for r in results if r["type"] == "entity"]
        assert len(entities) == 0


class TestTermExpansion:
    def test_returns_term_candidate(self, mock_engine) -> None:
        hit = {"code": "C34.1", "label": "Lung cancer", "final_score": 0.9}
        results = _expand_term_hit(mock_engine, hit)
        assert any(r["type"] == "term" for r in results)
        term = next(r for r in results if r["type"] == "term"
                    and r.get("relationship") is None)
        assert term["code"] == "C34.1"

    def test_expands_ancestry(self, mock_engine) -> None:
        mock_engine._run_query.return_value = [
            {"code": "C34", "label": "Malignant neoplasm of bronchus"},
        ]
        hit = {"code": "C34.1", "label": "Lung cancer", "final_score": 0.9}
        results = _expand_term_hit(mock_engine, hit)
        ancestors = [r for r in results if r["type"] == "ancestry"]
        assert len(ancestors) == 1
        assert ancestors[0]["code"] == "C34"

    def test_falls_back_to_name_when_no_code(self, mock_engine) -> None:
        hit = {"name": "lung_cancer", "final_score": 0.7}
        results = _expand_term_hit(mock_engine, hit)
        term = next(r for r in results if r["type"] == "term"
                    and r.get("relationship") is None)
        assert term["code"] == "lung_cancer"


class TestAliasExpansion:
    def test_returns_alias_candidate(self, mock_engine) -> None:
        hit = {"text": "BP", "parent_name": "Blood Pressure", "final_score": 0.8}
        results = _expand_alias_hit(mock_engine, hit)
        assert any(r["type"] == "alias" for r in results)
        alias = next(r for r in results if r["type"] == "alias")
        assert alias["text"] == "BP"
        assert alias["target"] == "Blood Pressure"

    def test_expands_target_as_entity(self, mock_engine) -> None:
        mock_engine._run_query.return_value = [
            {"table_name": "vitals", "schema_name": "clinical",
             "catalog": "cdm", "columns": []},
        ]
        hit = {"text": "BP", "parent_name": "Blood Pressure", "final_score": 0.8}
        results = _expand_alias_hit(mock_engine, hit)
        entities = [r for r in results if r["type"] == "entity"]
        assert len(entities) == 1
        assert entities[0]["name"] == "Blood Pressure"

    def test_no_expansion_when_no_target(self, mock_engine) -> None:
        hit = {"name": "BP", "final_score": 0.8}
        results = _expand_alias_hit(mock_engine, hit)
        entities = [r for r in results if r["type"] == "entity"]
        assert len(entities) == 0
