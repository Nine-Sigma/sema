import pytest
from unittest.mock import MagicMock, call

pytestmark = pytest.mark.unit

from sema.engine.embeddings import (
    EmbeddingEngine,
    build_embedding_text,
)


class TestEmbeddingTextConstruction:
    def test_entity_text(self):
        text = build_embedding_text(
            "entity", name="Cancer Diagnosis",
            description="Primary cancer diagnosis record",
        )
        assert "Cancer Diagnosis" in text
        assert "Primary cancer diagnosis record" in text

    def test_entity_without_description(self):
        text = build_embedding_text("entity", name="Test Entity")
        assert "Test Entity" in text

    def test_property_text(self):
        text = build_embedding_text(
            "property", name="Diagnosis Type",
            description="Cancer type classification",
        )
        assert "Diagnosis Type" in text

    def test_term_text(self):
        text = build_embedding_text("term", label="Colorectal Cancer")
        assert "Colorectal Cancer" in text

    def test_synonym_text(self):
        text = build_embedding_text("synonym", text="colon cancer")
        assert "colon cancer" in text

    def test_metric_text(self):
        text = build_embedding_text(
            "metric", name="Average Days to Surgery",
            description="Mean days between diagnosis and surgery",
        )
        assert "Average Days to Surgery" in text

    def test_transformation_text(self):
        text = build_embedding_text(
            "transformation", name="stg_cancer_diagnosis",
        )
        assert "stg_cancer_diagnosis" in text


class TestEmbeddingEngine:
    def test_embed_batch(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        engine = EmbeddingEngine(model=mock_model)
        results = engine.embed_batch(["text one", "text two"])
        assert len(results) == 2
        assert results[0] == [0.1, 0.2, 0.3]

    def test_embed_and_store(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]

        mock_loader = MagicMock()
        engine = EmbeddingEngine(model=mock_model, loader=mock_loader)

        engine.embed_and_store(
            label="Entity", match_prop="name",
            items=[{"name": "Cancer Diagnosis", "description": "Primary dx"}],
            text_fn=lambda item: build_embedding_text(
                "entity", **item,
            ),
        )

        mock_loader.set_embedding.assert_called_once()
        call_args = mock_loader.set_embedding.call_args
        assert call_args[1]["label"] == "Entity" or call_args[0][0] == "Entity"

    def test_create_vector_indexes(self):
        mock_loader = MagicMock()
        engine = EmbeddingEngine(model=MagicMock(), loader=mock_loader)
        engine.create_all_indexes(dimensions=384)
        # Should create indexes for Entity, Property, Term, Synonym,
        # Metric, Transformation
        assert mock_loader.create_vector_index.call_count >= 4
