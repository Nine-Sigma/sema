"""Tests for lexical retrieval, seed dedup, and artifact dedup."""

import pytest
from unittest.mock import MagicMock

from sema.pipeline.retrieval_utils import (
    _dedup_artifacts,
    _dedup_seeds,
    tokenize_query,
)

pytestmark = pytest.mark.unit


class TestTokenizeQuery:
    def test_basic_tokenization(self) -> None:
        tokens = tokenize_query("stage 3 colorectal cancer")
        assert "stage" in tokens
        assert "colorectal" in tokens
        assert "cancer" in tokens
        assert "3" not in tokens  # too short

    def test_drops_stop_words(self) -> None:
        tokens = tokenize_query("what are the most common mutations")
        assert "what" not in tokens
        assert "are" not in tokens
        assert "the" not in tokens
        assert "most" not in tokens
        assert "common" in tokens
        assert "mutations" in tokens

    def test_splits_on_punctuation(self) -> None:
        tokens = tokenize_query("cancer-diagnosis/staging")
        assert "cancer" in tokens
        assert "diagnosis" in tokens
        assert "staging" in tokens

    def test_empty_query(self) -> None:
        assert tokenize_query("") == []
        assert tokenize_query("   ") == []


class TestLexicalSearch:
    def test_term_matched_by_label(self) -> None:
        engine = MagicMock()
        node = {"label": "Genomic Mutation", "code": "GM01"}
        engine._run_query.return_value = [{"node": node}]

        from sema.pipeline.retrieval import RetrievalEngine
        real_engine = RetrievalEngine.__new__(RetrievalEngine)
        real_engine._driver = engine
        real_engine._embedder = None
        real_engine._run_query = engine._run_query

        hits = real_engine._lexical_search("mutations")
        term_hits = [h for h in hits if h.get("node_type") == "term"]
        assert len(term_hits) >= 1

    def test_alias_matched_by_text(self) -> None:
        engine = MagicMock()
        node = {"text": "CRC", "target_name": "Colorectal Cancer"}
        engine._run_query.return_value = [{"node": node}]

        from sema.pipeline.retrieval import RetrievalEngine
        real_engine = RetrievalEngine.__new__(RetrievalEngine)
        real_engine._driver = engine
        real_engine._embedder = None
        real_engine._run_query = engine._run_query

        hits = real_engine._lexical_search("CRC diagnosis")
        alias_hits = [h for h in hits if h.get("node_type") == "alias"]
        assert len(alias_hits) >= 1

    def test_no_lexical_matches(self) -> None:
        engine = MagicMock()
        engine._run_query.return_value = []

        from sema.pipeline.retrieval import RetrievalEngine
        real_engine = RetrievalEngine.__new__(RetrievalEngine)
        real_engine._driver = engine
        real_engine._embedder = None
        real_engine._run_query = engine._run_query

        hits = real_engine._lexical_search("xyznonexistent")
        assert hits == []

    def test_lexical_hit_outranks_weak_vector(self) -> None:
        from sema.pipeline.retrieval_utils import merge_and_rank_candidates

        candidates = [
            {"name": "A", "score": 0.5, "match_type": "vector",
             "confidence": 0.5, "node_type": "entity"},
            {"name": "A", "score": 1.0, "match_type": "lexical_exact",
             "confidence": 0.5, "node_type": "entity"},
        ]
        ranked = merge_and_rank_candidates(candidates)
        assert ranked[0]["match_type"] == "lexical_exact"


class TestSeedDedup:
    def test_same_entity_keeps_highest(self) -> None:
        seeds = [
            {"node_type": "entity", "name": "Cancer",
             "final_score": 0.5},
            {"node_type": "entity", "name": "Cancer",
             "final_score": 0.9},
        ]
        result = _dedup_seeds(seeds)
        assert len(result) == 1
        assert result[0]["final_score"] == 0.9

    def test_different_properties_same_name_distinct(self) -> None:
        seeds = [
            {"node_type": "property", "name": "status",
             "datasource_id": "ds1", "column_key": "t1.status",
             "final_score": 0.8},
            {"node_type": "property", "name": "status",
             "datasource_id": "ds1", "column_key": "t2.status",
             "final_score": 0.7},
        ]
        result = _dedup_seeds(seeds)
        assert len(result) == 2

    def test_property_without_scoped_identity_uses_fallback(
        self,
    ) -> None:
        seeds = [
            {"node_type": "property", "name": "status",
             "final_score": 0.8},
            {"node_type": "property", "name": "status",
             "final_score": 0.5},
        ]
        result = _dedup_seeds(seeds)
        assert len(result) == 1
        assert result[0]["final_score"] == 0.8


class TestArtifactDedup:
    def test_same_entity_from_two_paths_deduped(self) -> None:
        artifacts = [
            {"type": "entity", "catalog": "c", "schema": "s",
             "table": "t1", "confidence": 0.6},
            {"type": "entity", "catalog": "c", "schema": "s",
             "table": "t1", "confidence": 0.9},
        ]
        result = _dedup_artifacts(artifacts)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9

    def test_same_governed_value_deduped(self) -> None:
        artifacts = [
            {"type": "value", "table": "t1", "column": "c1",
             "code": "A", "confidence": 0.5},
            {"type": "value", "table": "t1", "column": "c1",
             "code": "A", "confidence": 0.8},
        ]
        result = _dedup_artifacts(artifacts)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.8

    def test_join_predicates_serialized_without_error(self) -> None:
        artifacts = [
            {"type": "join", "from_table": "t1", "to_table": "t2",
             "join_predicates": [{"left": "a", "right": "b"}],
             "confidence": 0.7},
            {"type": "join", "from_table": "t1", "to_table": "t2",
             "join_predicates": [{"left": "a", "right": "b"}],
             "confidence": 0.9},
        ]
        result = _dedup_artifacts(artifacts)
        assert len(result) == 1
