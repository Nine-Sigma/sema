"""Tests for context_utils helper functions."""

import pytest

from sema.models.context import SemanticCandidateSet
from sema.pipeline.context import prune_to_sco
from sema.pipeline.context_utils import (
    _apply_visibility_policy,
    _build_governed_values,
    _build_metrics,
    _parse_join_predicates,
)

pytestmark = pytest.mark.unit


class TestParseJoinPredicates:
    def test_empty_input(self) -> None:
        assert _parse_join_predicates(None) == []
        assert _parse_join_predicates("") == []

    def test_json_string_input(self) -> None:
        import json

        raw = json.dumps([{
            "left_table": "a", "left_column": "id",
            "right_table": "b", "right_column": "a_id",
        }])
        result = _parse_join_predicates(raw)
        assert len(result) == 1
        assert result[0].left_table == "a"
        assert result[0].right_column == "a_id"

    def test_invalid_json_string(self) -> None:
        assert _parse_join_predicates("{not valid json") == []

    def test_non_list_input(self) -> None:
        assert _parse_join_predicates(42) == []

    def test_list_with_dicts(self) -> None:
        raw = [
            {"left_table": "t1", "left_column": "c1",
             "right_table": "t2", "right_column": "c2"},
            {"left_table": "t3", "left_column": "c3",
             "right_table": "t4", "right_column": "c4"},
        ]
        result = _parse_join_predicates(raw)
        assert len(result) == 2

    def test_list_skips_non_dicts(self) -> None:
        raw = [{"left_table": "a"}, "not_a_dict", 42]
        result = _parse_join_predicates(raw)
        assert len(result) == 1


class TestApplyVisibilityPolicy:
    def test_rejects_rejected(self) -> None:
        candidates = [{"status": "rejected", "name": "x"}]
        assert _apply_visibility_policy(candidates, "nl2sql") == []

    def test_rejects_superseded(self) -> None:
        candidates = [{"status": "superseded", "name": "x"}]
        assert _apply_visibility_policy(candidates, "nl2sql") == []

    def test_includes_pinned(self) -> None:
        candidates = [{"status": "pinned", "name": "x"}]
        result = _apply_visibility_policy(candidates, "nl2sql")
        assert len(result) == 1

    def test_includes_accepted(self) -> None:
        candidates = [{"status": "accepted", "name": "x"}]
        result = _apply_visibility_policy(candidates, "nl2sql")
        assert len(result) == 1

    def test_auto_above_threshold(self) -> None:
        candidates = [
            {"status": "auto", "confidence": 0.9, "source": "llm"},
        ]
        result = _apply_visibility_policy(candidates, "nl2sql")
        assert len(result) == 1

    def test_auto_below_threshold(self) -> None:
        candidates = [
            {"status": "auto", "confidence": 0.3, "source": "llm"},
        ]
        result = _apply_visibility_policy(candidates, "nl2sql")
        assert len(result) == 0


    def test_low_confidence_semantic_artifact_filtered(self) -> None:
        candidates = [
            {"status": "auto", "confidence": 0.3,
             "confidence_policy": "semantic", "name": "x"},
        ]
        result = _apply_visibility_policy(candidates, "nl2sql")
        assert len(result) == 0

    def test_accepted_artifact_always_included(self) -> None:
        candidates = [
            {"status": "accepted", "confidence": 0.1,
             "confidence_policy": "semantic", "name": "x"},
        ]
        result = _apply_visibility_policy(candidates, "nl2sql")
        assert len(result) == 1

    def test_candidate_without_confidence_policy_uses_fallback(
        self,
    ) -> None:
        candidates = [
            {"status": "auto", "confidence": 0.6,
             "source": "structural", "name": "x"},
        ]
        result = _apply_visibility_policy(candidates, "nl2sql")
        # 0.6 >= 0.5 (structural threshold)
        assert len(result) == 1

    def test_structural_policy_lower_threshold(self) -> None:
        candidates = [
            {"status": "auto", "confidence": 0.55,
             "confidence_policy": "structural", "name": "x"},
        ]
        result = _apply_visibility_policy(candidates, "nl2sql")
        assert len(result) == 1


class TestBuildGovernedValues:
    def test_groups_by_property_column_table(self) -> None:
        cs = SemanticCandidateSet(
            query="test",
            candidates=[
                {"type": "value", "property_name": "Status",
                 "column": "status_cd", "table": "patients",
                 "code": "A", "label": "Active"},
                {"type": "value", "property_name": "Status",
                 "column": "status_cd", "table": "patients",
                 "code": "I", "label": "Inactive"},
                {"type": "entity", "name": "Patient"},
            ],
        )
        result = _build_governed_values(cs)
        assert len(result) == 1
        assert result[0].property_name == "Status"
        assert len(result[0].values) == 2


class TestBuildMetrics:
    def test_metric_candidate_survives_into_sco(self) -> None:
        cs = SemanticCandidateSet(
            query="test",
            candidates=[
                {"type": "entity", "name": "Hospital",
                 "table": "hospital", "schema": "s", "catalog": "c",
                 "confidence": 0.8, "source": "retrieval",
                 "columns": []},
                {"type": "metric", "name": "Avg LOS",
                 "description": "Average length of stay",
                 "formula": "AVG(los)", "confidence": 0.8,
                 "source": "retrieval",
                 "aggregates": ["LOS"], "filters": [],
                 "grains": ["Department"],
                 "status": "auto",
                 "confidence_policy": "semantic"},
            ],
        )
        sco = prune_to_sco(cs, consumer="nl2sql")
        assert len(sco.metrics) == 1
        assert sco.metrics[0].name == "Avg LOS"
        assert sco.metrics[0].aggregates == ["LOS"]
        assert sco.metrics[0].grains == ["Department"]

    def test_entity_description_preserved(self) -> None:
        cs = SemanticCandidateSet(
            query="test",
            candidates=[
                {"type": "entity", "name": "Cancer",
                 "table": "t1", "schema": "s", "catalog": "c",
                 "description": "Primary cancer records",
                 "confidence": 0.8, "source": "llm",
                 "columns": []},
            ],
        )
        sco = prune_to_sco(cs, consumer="nl2sql")
        assert sco.entities[0].description == "Primary cancer records"

    def test_property_semantic_type_preserved(self) -> None:
        cs = SemanticCandidateSet(
            query="test",
            candidates=[
                {"type": "entity", "name": "Cancer",
                 "table": "t1", "schema": "s", "catalog": "c",
                 "confidence": 0.8, "source": "llm",
                 "columns": [
                     {"property": "Stage", "column": "stage_cd",
                      "semantic_type": "categorical",
                      "vocabulary": "tnm_staging"},
                 ]},
            ],
        )
        sco = prune_to_sco(cs, consumer="nl2sql")
        prop = sco.entities[0].properties[0]
        assert prop.semantic_type == "categorical"
        assert prop.vocabulary == "tnm_staging"
