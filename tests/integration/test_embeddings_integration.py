import pytest

pytestmark = pytest.mark.integration

from sema.graph.loader import GraphLoader
from sema.engine.embeddings import EmbeddingEngine, build_embedding_text


class FakeEmbedder:
    """Simple fake embedder that returns deterministic vectors."""
    def encode(self, texts):
        return [[float(i) / 10.0] * 384 for i, _ in enumerate(texts)]


@pytest.fixture
def loaded_graph(clean_neo4j):
    loader = GraphLoader(clean_neo4j)
    loader.upsert_catalog("cdm")
    loader.upsert_schema("clinical", "cdm")
    loader.upsert_table("cancer_diagnosis", "clinical", "cdm")
    loader.upsert_entity("Cancer Diagnosis", description="Primary dx",
                        source="llm", confidence=0.8,
                        table_name="cancer_diagnosis", schema_name="clinical", catalog="cdm")
    loader.upsert_term("CRC", "Colorectal Cancer", source="llm", confidence=0.85)
    loader.upsert_synonym("colon cancer", parent_label=":Entity",
                         parent_name="Cancer Diagnosis", source="llm", confidence=0.8)
    return clean_neo4j, loader


class TestEmbeddingStorage:
    def test_embedding_stored_on_entity(self, loaded_graph):
        driver, loader = loaded_graph
        engine = EmbeddingEngine(model=FakeEmbedder(), loader=loader)

        engine.embed_and_store(
            label="Entity", match_prop="name",
            items=[{"name": "Cancer Diagnosis", "description": "Primary dx"}],
            text_fn=lambda item: build_embedding_text("entity", **item),
        )

        with driver.session() as s:
            node = s.run("MATCH (e:Entity {name: 'Cancer Diagnosis'}) RETURN e.embedding AS emb").single()
        assert node["emb"] is not None
        assert len(node["emb"]) == 384


class TestVectorIndexCreation:
    def test_create_indexes(self, loaded_graph):
        driver, loader = loaded_graph
        engine = EmbeddingEngine(model=FakeEmbedder(), loader=loader)
        engine.create_all_indexes(dimensions=384)

        with driver.session() as s:
            indexes = list(s.run("SHOW INDEXES YIELD name RETURN name"))
        index_names = {r["name"] for r in indexes}
        assert "entity_embeddings" in index_names
