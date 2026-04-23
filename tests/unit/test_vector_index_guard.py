"""Tests for the Neo4j vector-index dimension guard (tasks 4.5, 4.7)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sema.graph.vector_index_utils import (
    EmbeddingDimensionMismatchError,
    assert_retrieval_dim_matches,
    assert_write_dim_matches,
    get_declared_index_dimension,
    probe_embedder_dimension,
)

pytestmark = pytest.mark.unit


def _driver_returning(records_for_name: dict[str, list[dict]]):
    """Build a mock Neo4j driver whose SHOW VECTOR INDEXES returns per-name records."""
    driver = MagicMock()
    session = MagicMock()

    def run_side_effect(query, **params):
        name = params.get("name", "")
        rows = records_for_name.get(name, [])
        result = []
        for row in rows:
            record = MagicMock()
            record.get.side_effect = lambda key, r=row: r.get(key)
            result.append(record)
        return result

    session.run.side_effect = run_side_effect
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver


class TestGetDeclaredIndexDimension:
    def test_returns_dim_when_index_exists(self):
        driver = _driver_returning({
            "entity_embedding_index": [
                {"options": {"indexConfig": {"vector.dimensions": 1536}}},
            ],
        })
        assert get_declared_index_dimension(
            driver, "entity_embedding_index",
        ) == 1536

    def test_returns_none_when_index_missing(self):
        driver = _driver_returning({})
        assert get_declared_index_dimension(
            driver, "entity_embedding_index",
        ) is None

    def test_returns_none_on_driver_error(self):
        driver = MagicMock()
        driver.session.side_effect = RuntimeError("boom")
        assert get_declared_index_dimension(
            driver, "entity_embedding_index",
        ) is None


class TestAssertWriteDimMatches:
    def test_raises_on_mismatch(self):
        driver = _driver_returning({
            "entity_embedding_index": [
                {"options": {"indexConfig": {"vector.dimensions": 1536}}},
            ],
        })
        with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
            assert_write_dim_matches(
                driver,
                ["entity_embedding_index"],
                embedder_dim=1024,
                model_name="databricks-bge-large-en",
            )
        msg = str(exc_info.value)
        assert "1024" in msg
        assert "1536" in msg
        assert "entity_embedding_index" in msg
        assert "databricks-bge-large-en" in msg

    def test_passes_when_dims_match(self):
        driver = _driver_returning({
            "entity_embedding_index": [
                {"options": {"indexConfig": {"vector.dimensions": 1024}}},
            ],
        })
        assert_write_dim_matches(
            driver,
            ["entity_embedding_index"],
            embedder_dim=1024,
            model_name="databricks-bge-large-en",
        )

    def test_passes_when_no_index_exists(self):
        driver = _driver_returning({})
        assert_write_dim_matches(
            driver,
            ["entity_embedding_index"],
            embedder_dim=1024,
            model_name="databricks-bge-large-en",
        )


class TestAssertRetrievalDimMatches:
    def _embedder(self, dim: int) -> MagicMock:
        embedder = MagicMock(spec=["embed_query"])
        embedder.embed_query.return_value = [0.0] * dim
        return embedder

    def test_raises_on_mismatch(self):
        driver = _driver_returning({
            "entity_embedding_index": [
                {"options": {"indexConfig": {"vector.dimensions": 1536}}},
            ],
        })
        with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
            assert_retrieval_dim_matches(
                driver,
                self._embedder(1024),
                ["entity_embedding_index"],
                model_name="databricks-bge-large-en",
            )
        msg = str(exc_info.value)
        assert "retrieval" in msg.lower()
        assert "sema context" in msg or "sema query" in msg

    def test_passes_when_dims_match(self):
        driver = _driver_returning({
            "entity_embedding_index": [
                {"options": {"indexConfig": {"vector.dimensions": 1024}}},
            ],
        })
        assert_retrieval_dim_matches(
            driver,
            self._embedder(1024),
            ["entity_embedding_index"],
            model_name="databricks-bge-large-en",
        )

    def test_does_not_probe_when_no_index_exists(self):
        driver = _driver_returning({})
        embedder = MagicMock(spec=["embed_query"])
        embedder.embed_query.side_effect = AssertionError("should not probe")
        assert_retrieval_dim_matches(
            driver, embedder, ["entity_embedding_index"],
            model_name="anything",
        )
        embedder.embed_query.assert_not_called()


class TestProbeEmbedderDimension:
    def test_uses_sentence_transformers_dim(self):
        embedder = MagicMock(spec=["get_sentence_embedding_dimension"])
        embedder.get_sentence_embedding_dimension.return_value = 384
        assert probe_embedder_dimension(embedder) == 384

    def test_uses_embed_query(self):
        embedder = MagicMock(spec=["embed_query"])
        embedder.embed_query.return_value = [0.0] * 1024
        assert probe_embedder_dimension(embedder) == 1024

    def test_uses_embed_documents(self):
        embedder = MagicMock(spec=["embed_documents"])
        embedder.embed_documents.return_value = [[0.0] * 1536]
        assert probe_embedder_dimension(embedder) == 1536

    def test_uses_encode(self):
        embedder = MagicMock(spec=["encode"])
        embedder.encode.return_value = [[0.0] * 768]
        assert probe_embedder_dimension(embedder) == 768

    def test_raises_when_no_known_interface(self):
        embedder = MagicMock(spec=[])
        with pytest.raises(ValueError):
            probe_embedder_dimension(embedder)


class TestRetrievalEngineDimGuard:
    def test_retrieval_engine_raises_on_mismatch(self):
        from sema.pipeline.retrieval import RetrievalEngine
        driver = _driver_returning({
            "entity_embedding_index": [
                {"options": {"indexConfig": {"vector.dimensions": 1536}}},
            ],
        })
        embedder = MagicMock(spec=["embed_query"])
        embedder.embed_query.return_value = [0.0] * 1024
        with pytest.raises(EmbeddingDimensionMismatchError):
            RetrievalEngine(
                driver=driver,
                embedder=embedder,
                embedder_model_name="databricks-bge-large-en",
            )

    def test_retrieval_engine_passes_with_matching_dims(self):
        from sema.pipeline.retrieval import RetrievalEngine
        driver = _driver_returning({
            "entity_embedding_index": [
                {"options": {"indexConfig": {"vector.dimensions": 1024}}},
            ],
            "property_embedding_index": [],
            "term_embedding_index": [],
            "alias_embedding_index": [],
            "metric_embedding_index": [],
        })
        embedder = MagicMock(spec=["embed_query"])
        embedder.embed_query.return_value = [0.0] * 1024
        RetrievalEngine(
            driver=driver,
            embedder=embedder,
            embedder_model_name="databricks-bge-large-en",
        )

    def test_retrieval_engine_skips_guard_without_embedder(self):
        from sema.pipeline.retrieval import RetrievalEngine
        driver = _driver_returning({
            "entity_embedding_index": [
                {"options": {"indexConfig": {"vector.dimensions": 1536}}},
            ],
        })
        engine = RetrievalEngine(driver=driver, embedder=None)
        assert engine.vector_search("hello") == []
