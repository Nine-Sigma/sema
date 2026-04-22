import pytest
from unittest.mock import MagicMock, patch, call

pytestmark = pytest.mark.unit

from sema.engine.embeddings import (
    EmbeddingEngine,
    INDEX_CONFIGS,
    build_embedding_text,
)
from sema.models.config import (
    BuildConfig,
    EmbeddingConfig,
    QueryConfig,
)
from sema.pipeline.orchestrate_utils import (
    _compute_embeddings,
    _retrieve_context,
)


class TestEmbeddingComputation:
    def test_embed_and_store_called_for_all_node_types(self):
        mock_loader = MagicMock()
        # Use spec=[] to prevent MagicMock from auto-creating embed_documents
        mock_model = MagicMock(spec=["encode"])
        mock_model.encode.return_value = [[0.1, 0.2]]

        fake_nodes = {
            "Entity": [{"name": "Order", "description": "A purchase order"}],
            "Property": [{"name": "order_id", "entity_name": "Order"}],
            "Term": [{"code": "ACTIVE", "label": "Active"}],
            "Alias": [{"text": "purchase"}],
            "Metric": [{"name": "total_revenue", "description": "Sum of sales"}],
        }
        mock_loader.query_nodes_by_label.side_effect = lambda label: fake_nodes.get(label, [])

        engine = EmbeddingEngine(model=mock_model, loader=mock_loader)

        for label in [cfg[1] for cfg in INDEX_CONFIGS]:
            nodes = mock_loader.query_nodes_by_label(label)
            match_prop = "name"
            if label == "Term":
                match_prop = "code"
            elif label in ("Alias", "Synonym"):
                match_prop = "text"
            engine.embed_and_store(
                label=label,
                match_prop=match_prop,
                items=nodes,
                text_fn=lambda item, lbl=label: build_embedding_text(lbl.lower(), **item),
            )

        # One set_embedding call per node with data in fake_nodes
        expected = sum(len(v) for v in fake_nodes.values())
        assert mock_loader.set_embedding.call_count == expected

    def test_property_embedding_uses_composite_key(self):
        mock_loader = MagicMock()
        mock_model = MagicMock(spec=["encode"])
        mock_model.encode.return_value = [[0.1], [0.2]]

        property_nodes = [
            {"name": "status", "entity_name": "Order"},
            {"name": "status", "entity_name": "Shipment"},
        ]

        engine = EmbeddingEngine(model=mock_model, loader=mock_loader)
        engine.embed_and_store(
            label="Property",
            match_prop="name",
            items=property_nodes,
            text_fn=lambda item: build_embedding_text("property", **item),
        )

        set_calls = mock_loader.set_embedding.call_args_list
        assert len(set_calls) == 2

        match_values = [c.kwargs.get("match_value", c[1].get("match_value", None))
                        if c.kwargs else c[1]["match_value"]
                        for c in set_calls]
        assert match_values[0] == "status"
        assert match_values[1] == "status"

        mock_loader.set_property_embedding.assert_not_called()

    def test_per_label_match_property(self):
        mock_loader = MagicMock()
        mock_model = MagicMock(spec=["encode"])
        mock_model.encode.return_value = [[0.5]]

        engine = EmbeddingEngine(model=mock_model, loader=mock_loader)

        engine.embed_and_store(
            label="Term", match_prop="code",
            items=[{"code": "T001", "label": "Active"}],
            text_fn=lambda item: build_embedding_text("term", **item),
        )
        term_call = mock_loader.set_embedding.call_args
        assert term_call.kwargs["match_prop"] == "code"

        mock_loader.reset_mock()
        mock_model.encode.return_value = [[0.5]]
        engine.embed_and_store(
            label="Alias", match_prop="text",
            items=[{"text": "order"}],
            text_fn=lambda item: build_embedding_text("alias", **item),
        )
        alias_call = mock_loader.set_embedding.call_args
        assert alias_call.kwargs["match_prop"] == "text"

        mock_loader.reset_mock()
        mock_model.encode.return_value = [[0.5]]
        engine.embed_and_store(
            label="Entity", match_prop="name",
            items=[{"name": "Customer"}],
            text_fn=lambda item: build_embedding_text("entity", **item),
        )
        entity_call = mock_loader.set_embedding.call_args
        assert entity_call.kwargs["match_prop"] == "name"

    def test_embedding_batch_size_64(self):
        mock_loader = MagicMock()
        mock_model = MagicMock()

        items = [{"name": f"entity_{i}"} for i in range(100)]

        def fake_encode(texts):
            return [[0.1] for _ in texts]

        mock_model.encode.side_effect = fake_encode

        engine = EmbeddingEngine(model=mock_model, loader=mock_loader)
        batch_size = 64

        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            engine.embed_and_store(
                label="Entity",
                match_prop="name",
                items=batch,
                text_fn=lambda item: build_embedding_text("entity", **item),
            )

        for encode_call in mock_model.encode.call_args_list:
            texts = encode_call[0][0]
            assert len(texts) <= batch_size

    def test_embedding_dimensions_from_model(self):
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        engine = EmbeddingEngine(model=mock_model, loader=mock_loader)
        dim = mock_model.get_sentence_embedding_dimension()
        engine.create_all_indexes(dimensions=dim)

        for idx_call in mock_loader.create_vector_index.call_args_list:
            assert idx_call[1].get("dimensions", idx_call[0][2] if len(idx_call[0]) > 2 else None) == 384


class TestSkipEmbeddings:
    @patch("sema.pipeline.orchestrate_utils._get_embedder")
    def test_skip_embeddings_short_circuits(self, mock_get_embedder):
        """skip_embeddings=True must not touch the embedder or Neo4j indexes.

        Avoids probe-dim calls against a live provider and avoids the
        write-path dim guard firing during LLM-only eval runs.
        """
        config_skip = BuildConfig(
            catalog="test_catalog",
            schemas=["default"],
            skip_embeddings=True,
        )
        mock_loader = MagicMock()

        _compute_embeddings(config_skip, mock_loader)

        mock_get_embedder.assert_not_called()
        mock_loader.create_vector_index.assert_not_called()
        mock_loader.set_embedding.assert_not_called()

    @patch("sema.pipeline.orchestrate_utils._get_embedder")
    def test_no_skip_invokes_embedder(self, mock_get_embedder):
        config = BuildConfig(catalog="test_catalog", schemas=["default"])
        mock_loader = MagicMock()
        _compute_embeddings(config, mock_loader)
        mock_get_embedder.assert_called_once()


class TestQueryEmbeddingConfig:
    def test_query_config_has_embedding_field(self):
        config = QueryConfig(
            question="test question",
            embedding=EmbeddingConfig(
                provider="sentence-transformers",
                model="all-MiniLM-L6-v2",
            ),
        )
        assert hasattr(config, "embedding")
        assert isinstance(config.embedding, EmbeddingConfig)
        assert config.embedding.provider == "sentence-transformers"

    def test_retrieve_context_uses_embedding_config(self):
        embedding_cfg = EmbeddingConfig(
            provider="sentence-transformers",
            model="all-MiniLM-L6-v2",
        )
        config = QueryConfig(
            question="What is the total revenue?",
            embedding=embedding_cfg,
        )

        assert hasattr(config, "embedding")
        assert isinstance(config.embedding, EmbeddingConfig)

        with patch("sema.pipeline.orchestrate_utils._get_neo4j_driver") as mock_get_driver, \
             patch("sema.cli_factories._get_embedder") as mock_get_embedder:
            mock_driver = MagicMock()
            mock_get_driver.return_value = mock_driver
            mock_embedder = MagicMock()
            mock_get_embedder.return_value = mock_embedder

            mock_retrieval_engine = MagicMock()
            mock_retrieval_engine.retrieve.return_value = {}

            with patch("sema.pipeline.orchestrate_utils.RetrievalEngine", return_value=mock_retrieval_engine), \
                 patch("sema.pipeline.orchestrate_utils.prune_to_sco", return_value={}):
                _retrieve_context(config)

            mock_get_embedder.assert_called_once_with(config.embedding)
