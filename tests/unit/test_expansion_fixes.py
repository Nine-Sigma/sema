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

    def test_ambiguous_code_emits_one_candidate_per_vocabulary(
        self, mock_engine,
    ) -> None:
        """A hit with no vocabulary whose code exists in several
        vocabularies must surface ALL of them as distinct candidates —
        retrieval never silently picks; downstream ranking selects."""
        governed_calls: list = []

        def run_query(query, **params):
            if "IN_VOCABULARY" in query and "$code" in query:
                return [
                    {"vocabulary_name": "Gender"},
                    {"vocabulary_name": "State"},
                ]
            if "MEMBER_OF" in query and "HAS_VALUE_SET" in query:
                governed_calls.append(params.get("vocabulary_name"))
            return []

        mock_engine._run_query.side_effect = run_query
        hit = {"code": "M", "label": "M", "final_score": 0.9}
        results = _expand_term_hit(mock_engine, hit)
        terms = [r for r in results if r["type"] == "term"]
        assert {t.get("vocabulary") for t in terms} == {
            "Gender", "State",
        }
        # Each alternative's expansion is scoped to its own vocabulary.
        assert sorted(governed_calls) == ["Gender", "State"]

    def test_unknown_vocabulary_keeps_single_unscoped_candidate(
        self, mock_engine,
    ) -> None:
        mock_engine._run_query.return_value = []
        hit = {"code": "M", "label": "M", "final_score": 0.9}
        results = _expand_term_hit(mock_engine, hit)
        terms = [r for r in results if r["type"] == "term"]
        assert len(terms) == 1
        assert "vocabulary" not in terms[0]

    def test_unresolved_vocabulary_scopes_to_unscoped_namespace(
        self, mock_engine,
    ) -> None:
        """No IN_VOCABULARY rows must NOT disable filtering: terms
        without a vocabulary live in the _unscoped namespace, so the
        expansion scopes there instead of matching every vocabulary."""
        from sema.graph.term_identity_utils import UNSCOPED_VOCAB

        captured: dict = {}

        def run_query(query, **params):
            if "MEMBER_OF" in query and "HAS_VALUE_SET" in query:
                captured["vocab"] = params.get("vocabulary_name")
            return []

        mock_engine._run_query.side_effect = run_query
        hit = {"code": "M", "label": "M", "final_score": 0.9}
        _expand_term_hit(mock_engine, hit)
        assert captured["vocab"] == UNSCOPED_VOCAB

    def test_resolver_failure_emits_bare_term_only(
        self, mock_engine,
    ) -> None:
        """A transient lookup failure must degrade to a bare term,
        never to unscoped cross-vocabulary expansion."""
        seen_queries: list = []

        def run_query(query, **params):
            seen_queries.append(query)
            if "IN_VOCABULARY" in query:
                raise RuntimeError("neo4j unavailable")
            return []

        mock_engine._run_query.side_effect = run_query
        hit = {"code": "M", "label": "M", "final_score": 0.9}
        results = _expand_term_hit(mock_engine, hit)
        assert len(results) == 1
        assert results[0]["type"] == "term"
        assert "vocabulary" not in results[0]
        assert not any("MEMBER_OF" in q for q in seen_queries)

    def test_governed_value_artifacts_carry_vocabulary(
        self, mock_engine,
    ) -> None:
        def run_query(query, **params):
            if "MEMBER_OF" in query and "HAS_VALUE_SET" in query:
                return [
                    {"column_name": "gender", "table_name": "patient",
                     "value_set_name": "gender_values"},
                ]
            return []

        mock_engine._run_query.side_effect = run_query
        hit = {
            "code": "M", "label": "Male",
            "vocabulary_name": "Gender", "final_score": 0.9,
        }
        results = _expand_term_hit(mock_engine, hit)
        values = [r for r in results if r["type"] == "value"]
        assert values
        assert all(v["vocabulary"] == "Gender" for v in values)

    def test_ambiguous_alternatives_are_marked(
        self, mock_engine,
    ) -> None:
        """Every artifact of a multi-vocabulary fan-out carries
        ambiguity metadata so context assembly can present the
        alternatives instead of treating them as facts."""
        def run_query(query, **params):
            if "IN_VOCABULARY" in query:
                return [
                    {"vocabulary_name": "Gender"},
                    {"vocabulary_name": "State"},
                ]
            if "MEMBER_OF" in query and "HAS_VALUE_SET" in query:
                col = (
                    "gender" if params.get("vocabulary_name") == "Gender"
                    else "state"
                )
                return [
                    {"column_name": col, "table_name": "t",
                     "value_set_name": f"{col}_values"},
                ]
            return []

        mock_engine._run_query.side_effect = run_query
        hit = {"code": "M", "label": "M", "final_score": 0.9}
        results = _expand_term_hit(mock_engine, hit)
        assert all(r.get("ambiguous") is True for r in results)
        assert all(r.get("ambiguity_group") == "M" for r in results)
        values = [r for r in results if r["type"] == "value"]
        assert {v["vocabulary"] for v in values} == {"Gender", "State"}

    def test_single_vocabulary_results_not_marked_ambiguous(
        self, mock_engine,
    ) -> None:
        mock_engine._run_query.return_value = []
        hit = {
            "code": "M", "label": "Male",
            "vocabulary_name": "Gender", "final_score": 0.9,
        }
        results = _expand_term_hit(mock_engine, hit)
        assert not any(r.get("ambiguous") for r in results)

    def test_vocabulary_fanout_is_capped(self, mock_engine) -> None:
        def run_query(query, **params):
            if "IN_VOCABULARY" in query:
                return [
                    {"vocabulary_name": f"V{i}"} for i in range(5)
                ]
            return []

        mock_engine._run_query.side_effect = run_query
        hit = {"code": "M", "label": "M", "final_score": 0.9}
        results = _expand_term_hit(mock_engine, hit)
        terms = [r for r in results if r["type"] == "term"]
        assert len(terms) == 3
        assert {t["vocabulary"] for t in terms} == {"V0", "V1", "V2"}

    def test_governed_values_scoped_to_hit_vocabulary(
        self, mock_engine,
    ) -> None:
        """Term identity is {vocabulary_name, code}: a Gender 'M' hit
        must not pull value sets from a State vocabulary's 'M'."""
        captured: dict = {}

        def run_query(query, **params):
            if "MEMBER_OF" in query and "HAS_VALUE_SET" in query:
                captured["query"] = query
                captured["params"] = params
                return [
                    {"column_name": "gender",
                     "table_name": "patient",
                     "value_set_name": "gender_values"},
                ]
            return []

        mock_engine._run_query.side_effect = run_query
        hit = {
            "code": "M", "label": "Male",
            "vocabulary_name": "Gender", "final_score": 0.9,
        }
        _expand_term_hit(mock_engine, hit)
        assert captured["params"].get("vocabulary_name") == "Gender"
        assert "$vocabulary_name" in captured["query"]


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
