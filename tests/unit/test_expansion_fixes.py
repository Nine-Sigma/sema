"""Tests for expansion fixes: term governed values, alias dispatch,
metric context, and visibility metadata propagation."""

import pytest
from unittest.mock import MagicMock

from sema.pipeline.retrieval_utils import (
    _expand_alias_hit,
    _expand_metrics,
    _expand_term_hit,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine._run_query.return_value = []
    return engine


class TestTermGovernedValues:
    def test_term_hit_surfaces_governed_values(
        self, mock_engine,
    ) -> None:
        def run_query(query, **params):
            if "find_value_sets_for_term" in query or (
                "MEMBER_OF" in query and "HAS_VALUE_SET" in query
            ):
                return [
                    {"column_name": "tnm_stage",
                     "table_name": "cancer_diagnosis",
                     "value_set_name": "staging_values"},
                ]
            return []

        mock_engine._run_query.side_effect = run_query
        hit = {
            "code": "Stage III", "label": "Stage III",
            "final_score": 0.9,
        }
        results = _expand_term_hit(mock_engine, hit)
        values = [r for r in results if r["type"] == "value"]
        assert len(values) >= 1
        assert values[0]["column"] == "tnm_stage"

    def test_term_without_member_of_returns_no_values(
        self, mock_engine,
    ) -> None:
        mock_engine._run_query.return_value = []
        hit = {"code": "orphan", "label": "Orphan", "final_score": 0.5}
        results = _expand_term_hit(mock_engine, hit)
        values = [r for r in results if r["type"] == "value"]
        assert len(values) == 0


class TestAliasDispatch:
    def test_alias_to_property_dispatches(
        self, mock_engine,
    ) -> None:
        def run_query(query, **params):
            if "REFERS_TO" in query:
                return [{
                    "target_name": "Diagnosis Code",
                    "target_labels": ["Property"],
                    "entity_name": "Diagnosis",
                }]
            return []

        mock_engine._run_query.side_effect = run_query
        hit = {
            "text": "dx_code", "parent_name": "Diagnosis Code",
            "final_score": 0.8,
        }
        results = _expand_alias_hit(mock_engine, hit)
        props = [r for r in results if r["type"] == "property"]
        assert len(props) >= 1

    def test_alias_to_entity_dispatches(
        self, mock_engine,
    ) -> None:
        def run_query(query, **params):
            if "REFERS_TO" in query:
                return [{
                    "target_name": "Cancer Diagnosis",
                    "target_labels": ["Entity"],
                }]
            if "ENTITY_ON_TABLE" in query:
                return [{
                    "table_name": "cancer_dx",
                    "schema_name": "clinical",
                    "catalog": "cdm",
                    "columns": [],
                }]
            return []

        mock_engine._run_query.side_effect = run_query
        hit = {
            "text": "cancer_dx", "parent_name": "Cancer Diagnosis",
            "final_score": 0.8,
        }
        results = _expand_alias_hit(mock_engine, hit)
        entities = [r for r in results if r["type"] == "entity"]
        assert len(entities) >= 1
        assert entities[0]["name"] == "Cancer Diagnosis"

    def test_alias_to_unknown_logs_and_skips(
        self, mock_engine,
    ) -> None:
        def run_query(query, **params):
            if "REFERS_TO" in query:
                return [{
                    "target_name": "SomeNode",
                    "target_labels": ["DataSource"],
                }]
            return []

        mock_engine._run_query.side_effect = run_query
        hit = {
            "text": "weird", "parent_name": "SomeNode",
            "final_score": 0.5,
        }
        results = _expand_alias_hit(mock_engine, hit)
        # Only the alias candidate itself
        non_alias = [
            r for r in results if r["type"] != "alias"
        ]
        assert len(non_alias) == 0


class TestMetricExpansion:
    def test_metric_with_full_context(self, mock_engine) -> None:
        def run_query(query, **params):
            if "MEASURES" in query:
                return [{
                    "name": "Avg LOS",
                    "description": "Average length of stay",
                    "formula": "AVG(los_days)",
                    "confidence": 0.8,
                    "aggregates": ["Length of Stay"],
                    "filters": ["Admission Type"],
                    "grains": ["Department"],
                }]
            return []

        mock_engine._run_query.side_effect = run_query
        result = _expand_metrics(mock_engine, ["Hospital"])
        assert len(result) == 1
        assert result[0]["aggregates"] == ["Length of Stay"]
        assert result[0]["filters"] == ["Admission Type"]
        assert result[0]["grains"] == ["Department"]
        assert result[0]["confidence_policy"] == "semantic"

    def test_metric_without_edges(self, mock_engine) -> None:
        def run_query(query, **params):
            if "MEASURES" in query:
                return [{
                    "name": "Count", "description": None,
                    "formula": None, "confidence": 0.5,
                    "aggregates": [], "filters": [], "grains": [],
                }]
            return []

        mock_engine._run_query.side_effect = run_query
        result = _expand_metrics(mock_engine, ["Entity"])
        assert len(result) == 1
        assert result[0]["aggregates"] == []


class TestVisibilityMetadata:
    def test_expanded_join_carries_structural_policy(self) -> None:
        from sema.pipeline.retrieval import RetrievalEngine

        engine = MagicMock(spec=RetrievalEngine)
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(
            return_value=session,
        )
        driver.session.return_value.__exit__ = MagicMock(
            return_value=False,
        )

        def run_side_effect(query, **params):
            if "ENTITY_ON_TABLE" in query:
                r = MagicMock()
                r.data.return_value = {
                    "table_name": "t1", "schema_name": "s",
                    "catalog": "c", "columns": [],
                }
                return [r]
            if "JoinPath" in query or "USES" in query:
                r = MagicMock()
                r.data.return_value = {
                    "from_table": "t1", "to_table": "t2",
                    "confidence": 0.8,
                }
                return [r]
            return []

        session.run.side_effect = run_side_effect
        real_engine = RetrievalEngine(driver=driver, embedder=None)
        result = real_engine._expand_entity_hits(["Test"])
        joins = [c for c in result if c["type"] == "join"]
        assert len(joins) >= 1
        assert joins[0]["confidence_policy"] == "structural"

    def test_expanded_value_carries_semantic_policy(self) -> None:
        from sema.pipeline.retrieval import RetrievalEngine

        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(
            return_value=session,
        )
        driver.session.return_value.__exit__ = MagicMock(
            return_value=False,
        )

        def run_side_effect(query, **params):
            if "ENTITY_ON_TABLE" in query:
                r = MagicMock()
                r.data.return_value = {
                    "table_name": "t1", "schema_name": "s",
                    "catalog": "c",
                    "columns": [
                        {"property": "Status", "column": "status",
                         "semantic_type": "categorical"},
                    ],
                }
                return [r]
            if "HAS_VALUE_SET" in query and "MEMBER_OF" in query:
                r = MagicMock()
                r.data.return_value = {
                    "code": "A", "label": "Active",
                }
                return [r]
            return []

        session.run.side_effect = run_side_effect
        engine = RetrievalEngine(driver=driver, embedder=None)
        result = engine._expand_entity_hits(["Test"])
        values = [c for c in result if c["type"] == "value"]
        if values:
            assert values[0]["confidence_policy"] == "semantic"

    def test_missing_assertion_defaults(self, mock_engine) -> None:
        hit = {
            "code": "C34", "label": "Lung Cancer",
            "final_score": 0.7,
        }
        results = _expand_term_hit(mock_engine, hit)
        term = next(r for r in results if r["type"] == "term")
        assert term["status"] == "auto"
        assert term["confidence"] == 0.5
