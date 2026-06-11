import hashlib
import pytest
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit

from sema.engine.embeddings import build_embedding_text
from sema.engine.embedding_utils import (
    description_hash,
    needs_reembedding,
    select_stale_nodes,
)
from sema.graph.loader import GraphLoader
from sema.pipeline.orchestrate_utils import _embed_label_nodes


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


class TestDescriptionHash:
    def test_hash_is_sha256_of_text(self):
        text = "Order - A purchase order"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert description_hash(text) == expected

    def test_hash_changes_with_text(self):
        assert description_hash("a") != description_hash("b")

    def test_hash_is_stable(self):
        assert description_hash("x") == description_hash("x")


class TestNeedsReembedding:
    def test_missing_embedding_needs_reembed(self):
        node = {"name": "Order", "description": "desc"}
        assert needs_reembedding(node, "Order - desc") is True

    def test_empty_embedding_needs_reembed(self):
        node = {"embedding": [], "description_hash": description_hash("t")}
        assert needs_reembedding(node, "t") is True

    def test_changed_hash_needs_reembed(self):
        node = {"embedding": [0.1], "description_hash": description_hash("old")}
        assert needs_reembedding(node, "new") is True

    def test_present_embedding_missing_hash_needs_reembed(self):
        node = {"embedding": [0.1]}
        assert needs_reembedding(node, "t") is True

    def test_unchanged_hash_skips(self):
        text = "Order - desc"
        node = {"embedding": [0.1], "description_hash": description_hash(text)}
        assert needs_reembedding(node, text) is False


class TestSelectStaleNodes:
    def test_pairs_only_stale(self):
        text_a = "A"
        nodes = [
            {"embedding": [0.1], "description_hash": description_hash(text_a)},
            {"name": "B"},
        ]
        stale = select_stale_nodes(nodes, [text_a, "B"])
        assert len(stale) == 1
        assert stale[0][0] is nodes[1]
        assert stale[0][1] == "B"

    def test_all_fresh_returns_empty(self):
        nodes = [{"embedding": [0.1], "description_hash": description_hash("x")}]
        assert select_stale_nodes(nodes, ["x"]) == []


class TestEmbedLabelNodesChangeDetection:
    def test_unchanged_nodes_not_reembedded(self):
        text = build_embedding_text("entity", name="Order", description="d")
        nodes = [
            {
                "name": "Order",
                "description": "d",
                "embedding": [0.1],
                "description_hash": description_hash(text),
            }
        ]
        engine = MagicMock()
        loader = MagicMock()
        _embed_label_nodes(engine, loader, "Entity", "name", nodes)
        engine.embed_batch.assert_not_called()
        loader.set_embedding.assert_not_called()

    def test_changed_node_reembedded_and_writes_hash(self):
        nodes = [
            {
                "name": "Order",
                "description": "new",
                "embedding": [0.1],
                "description_hash": "stale",
            }
        ]
        engine = MagicMock()
        engine.embed_batch.return_value = [[0.9]]
        loader = MagicMock()
        _embed_label_nodes(engine, loader, "Entity", "name", nodes)
        engine.embed_batch.assert_called_once()
        loader.set_embedding.assert_called_once()
        kwargs = loader.set_embedding.call_args.kwargs
        new_text = build_embedding_text("entity", name="Order", description="new")
        assert kwargs["description_hash"] == description_hash(new_text)

    def test_missing_embedding_reembedded(self):
        nodes = [{"name": "Order", "description": "d"}]
        engine = MagicMock()
        engine.embed_batch.return_value = [[0.9]]
        loader = MagicMock()
        _embed_label_nodes(engine, loader, "Entity", "name", nodes)
        loader.set_embedding.assert_called_once()

    def test_property_composite_key_writes_hash(self):
        nodes = [{"name": "status", "entity_name": "Order"}]
        engine = MagicMock()
        engine.embed_batch.return_value = [[0.9]]
        loader = MagicMock()
        _embed_label_nodes(
            engine, loader, "Property", ("name", "entity_name"), nodes,
        )
        loader.set_property_embedding.assert_called_once()
        assert "description_hash" in loader.set_property_embedding.call_args.kwargs

    def test_mixed_only_stale_embedded(self):
        fresh_text = build_embedding_text("entity", name="A", description="d")
        nodes = [
            {
                "name": "A",
                "description": "d",
                "embedding": [0.1],
                "description_hash": description_hash(fresh_text),
            },
            {"name": "B", "description": "d2"},
        ]
        engine = MagicMock()
        engine.embed_batch.return_value = [[0.9]]
        loader = MagicMock()
        _embed_label_nodes(engine, loader, "Entity", "name", nodes)
        engine.embed_batch.assert_called_once()
        embedded_texts = engine.embed_batch.call_args[0][0]
        assert embedded_texts == [
            build_embedding_text("entity", name="B", description="d2")
        ]
        loader.set_embedding.assert_called_once()
        assert loader.set_embedding.call_args.kwargs["match_value"] == "B"


class TestLoaderWritesDescriptionHash:
    def test_set_embedding_writes_hash_in_same_statement(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        loader.set_embedding(
            label="Entity", match_prop="name", match_value="Order",
            embedding=[0.1], description_hash="abc123",
        )
        cypher = session.run.call_args[0][0]
        params = session.run.call_args[1]
        assert "n.description_hash = $description_hash" in cypher
        assert "n.embedding = $embedding" in cypher
        assert params["description_hash"] == "abc123"

    def test_set_property_embedding_writes_hash_in_same_statement(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        loader.set_property_embedding(
            name="status", entity_name="Order",
            embedding=[0.1], description_hash="def456",
        )
        cypher = session.run.call_args[0][0]
        params = session.run.call_args[1]
        assert "n.description_hash = $description_hash" in cypher
        assert params["description_hash"] == "def456"
