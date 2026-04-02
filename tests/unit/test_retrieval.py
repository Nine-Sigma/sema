import pytest
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit

from sema.pipeline.retrieval import (
    RetrievalEngine,
    merge_and_rank_candidates,
)
from sema.pipeline.context import (
    prune_to_sco,
)
from sema.models.context import SemanticCandidateSet


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.encode.return_value = [[0.1, 0.2, 0.3]]
    return embedder


class TestHybridRetrieval:
    def test_vector_search_returns_candidates(self, mock_driver, mock_embedder):
        driver, session = mock_driver
        session.run.return_value = [
            MagicMock(data=lambda: {"node": {"name": "Cancer Diagnosis"}, "score": 0.94}),
            MagicMock(data=lambda: {"node": {"name": "Stage III"}, "score": 0.88}),
        ]
        engine = RetrievalEngine(driver=driver, embedder=mock_embedder)
        candidates = engine.vector_search("stage 3 colorectal")
        assert len(candidates) >= 1

    def test_graph_expansion_adds_related_nodes(self, mock_driver, mock_embedder):
        driver, session = mock_driver

        def make_vector_result(name, score):
            r = MagicMock()
            r.data.return_value = {"node": {"name": name}, "score": score}
            return r

        def make_physical_result():
            r = MagicMock()
            r.data.return_value = {
                "table_name": "cancer_diagnosis",
                "schema_name": "clinical",
                "catalog": "cdm",
                "columns": [{"property": "Diagnosis Type", "column": "dx_type_cd",
                            "data_type": "STRING", "semantic_type": "categorical"}],
            }
            return r

        # 5 vector searches (one per index) + expansion queries
        session.run.side_effect = [
            # entity_embedding_index vector search
            [make_vector_result("Cancer Diagnosis", 0.9)],
            # property_embedding_index
            [],
            # term_embedding_index
            [],
            # alias_embedding_index
            [],
            # metric_embedding_index
            [],
            # physical mapping for "Cancer Diagnosis"
            [make_physical_result()],
            # join paths
            [],
            # value set expansion
            [],
            # metrics
            [],
            # ancestry
            [],
        ]
        engine = RetrievalEngine(driver=driver, embedder=mock_embedder)
        candidate_set = engine.retrieve("stage 3 colorectal", top_k=5)
        assert isinstance(candidate_set, SemanticCandidateSet)
        assert len(candidate_set.candidates) > 0

    def test_lexical_matching_boosts_exact(self, mock_driver, mock_embedder):
        driver, session = mock_driver
        session.run.return_value = []
        engine = RetrievalEngine(driver=driver, embedder=mock_embedder)
        candidates = [
            {"name": "Stage III", "score": 0.7, "match_type": "vector"},
            {"name": "stage iii", "score": 0.5, "match_type": "lexical_exact"},
        ]
        ranked = merge_and_rank_candidates(candidates)
        # Lexical exact should rank higher
        assert ranked[0]["name"] == "stage iii"


class TestMultiSignalRanking:
    def test_confidence_weighting(self):
        candidates = [
            {"name": "A", "score": 0.8, "confidence": 0.5, "match_type": "vector"},
            {"name": "B", "score": 0.7, "confidence": 1.0, "match_type": "vector"},
        ]
        ranked = merge_and_rank_candidates(candidates)
        # B has higher confidence, should rank higher
        assert ranked[0]["name"] == "B"

    def test_lexical_beats_vector(self):
        candidates = [
            {"name": "A", "score": 0.9, "confidence": 0.8, "match_type": "vector"},
            {"name": "B", "score": 0.6, "confidence": 0.8, "match_type": "lexical_exact"},
        ]
        ranked = merge_and_rank_candidates(candidates)
        assert ranked[0]["name"] == "B"


class TestContextPruning:
    def test_nl2sql_pruning(self):
        candidate_set = SemanticCandidateSet(
            query="test",
            candidates=[
                {"type": "entity", "name": "Cancer Diagnosis",
                 "table": "cancer_diagnosis", "schema": "clinical", "catalog": "cdm",
                 "description": "Primary dx", "confidence": 0.8, "source": "llm",
                 "columns": [{"property": "Stage", "column": "tnm_stage",
                            "data_type": "STRING", "semantic_type": "categorical"}]},
                {"type": "join", "from_table": "cancer_diagnosis",
                 "to_table": "cancer_surgery", "on_column": "patient_id",
                 "cardinality": "one-to-many", "confidence": 0.8,
                 "from_schema": "clinical", "from_catalog": "cdm",
                 "to_schema": "clinical", "to_catalog": "cdm"},
                {"type": "value", "property_name": "Stage",
                 "column": "tnm_stage", "table": "cancer_diagnosis",
                 "code": "Stage III", "label": "Stage III"},
            ],
        )
        sco = prune_to_sco(candidate_set, consumer="nl2sql")
        assert sco.consumer == "nl2sql"
        assert len(sco.entities) >= 1
        assert len(sco.physical_assets) >= 1

    def test_discovery_pruning_keeps_descriptions(self):
        candidate_set = SemanticCandidateSet(
            query="test",
            candidates=[
                {"type": "entity", "name": "Cancer Diagnosis",
                 "table": "cancer_diagnosis", "schema": "clinical", "catalog": "cdm",
                 "description": "Primary diagnosis record", "confidence": 0.8,
                 "source": "llm", "columns": []},
            ],
        )
        sco = prune_to_sco(candidate_set, consumer="discovery")
        assert sco.consumer == "discovery"
        assert sco.entities[0].description == "Primary diagnosis record"

    def test_context_budget(self):
        candidates = [
            {"type": "entity", "name": f"Entity {i}",
             "table": f"tbl{i}", "schema": "s", "catalog": "c",
             "description": f"desc {i}", "confidence": 0.5 + i * 0.01,
             "source": "llm", "columns": []}
            for i in range(20)
        ]
        candidate_set = SemanticCandidateSet(query="test", candidates=candidates)
        sco = prune_to_sco(candidate_set, consumer="nl2sql", max_entities=5)
        assert len(sco.entities) <= 5


class TestIndexToNodeType:
    def test_known_indices(self, mock_driver):
        driver, _ = mock_driver
        engine = RetrievalEngine(driver=driver, embedder=None)
        assert engine._index_to_node_type("entity_embedding_index") == "entity"
        assert engine._index_to_node_type("property_embedding_index") == "property"
        assert engine._index_to_node_type("term_embedding_index") == "term"
        assert engine._index_to_node_type("alias_embedding_index") == "alias"
        assert engine._index_to_node_type("metric_embedding_index") == "metric"

    def test_unknown_index(self, mock_driver):
        driver, _ = mock_driver
        engine = RetrievalEngine(driver=driver, embedder=None)
        assert engine._index_to_node_type("something_else") == "unknown"


class TestDispatchNonEntityHit:
    def test_metric_hit(self, mock_driver):
        driver, _ = mock_driver
        engine = RetrievalEngine(driver=driver, embedder=None)
        hit = {"node_type": "metric", "name": "revenue", "final_score": 0.9}
        result = engine._dispatch_non_entity_hit(hit)
        assert len(result) == 1
        assert result[0]["type"] == "metric"
        assert result[0]["name"] == "revenue"

    def test_unknown_type_returns_empty(self, mock_driver):
        driver, _ = mock_driver
        engine = RetrievalEngine(driver=driver, embedder=None)
        hit = {"node_type": "unknown", "name": "x"}
        assert engine._dispatch_non_entity_hit(hit) == []

    def test_property_without_name_returns_empty(self, mock_driver):
        driver, _ = mock_driver
        engine = RetrievalEngine(driver=driver, embedder=None)
        hit = {"node_type": "property", "name": ""}
        assert engine._dispatch_non_entity_hit(hit) == []


class TestRetrieveNonEntityHits:
    def test_non_entity_hits_dispatched(self, mock_driver, mock_embedder):
        driver, session = mock_driver

        def make_result(name, score):
            r = MagicMock()
            r.data.return_value = {
                "node": {"name": name}, "score": score,
            }
            return r

        session.run.side_effect = [
            [],  # entity index
            [make_result("diagnosis_code", 0.85)],  # property index
            [],  # term index
            [],  # alias index
            [],  # metric index
        ]
        engine = RetrievalEngine(driver=driver, embedder=mock_embedder)
        result = engine.retrieve("diagnosis codes", top_k=5)
        assert isinstance(result, SemanticCandidateSet)

    def test_vector_search_without_embedder(self, mock_driver):
        driver, _ = mock_driver
        engine = RetrievalEngine(driver=driver, embedder=None)
        assert engine.vector_search("test") == []

    def test_vector_search_encode_fallback(self, mock_driver):
        driver, session = mock_driver
        session.run.return_value = []
        embedder = MagicMock(spec=["encode"])
        embedder.encode.return_value = [[0.1, 0.2]]
        engine = RetrievalEngine(driver=driver, embedder=embedder)
        engine.vector_search("test")
        embedder.encode.assert_called_once()


class TestExpandEntityHits:
    def test_includes_joins_values_metrics(self, mock_driver):
        driver, session = mock_driver

        def run_side_effect(query, **params):
            r = MagicMock()
            if "ENTITY_ON_TABLE" in query:
                r.data.return_value = {
                    "table_name": "tbl", "schema_name": "sch",
                    "catalog": "cat", "columns": [
                        {"property": "Status", "column": "status",
                         "data_type": "STRING", "semantic_type": "categorical"},
                    ],
                }
                return [r]
            if "JoinPath" in query or "USES" in query:
                r.data.return_value = {
                    "from_table": "tbl", "to_table": "tbl2",
                    "on_column": "id", "confidence": 0.8,
                }
                return [r]
            if "MEMBER_OF" in query:
                r.data.return_value = {
                    "property": "Status", "column": "status",
                    "table": "tbl", "code": "A", "label": "Active",
                }
                return [r]
            if "MEASURES" in query:
                r.data.return_value = {
                    "name": "total", "description": "sum",
                }
                return [r]
            return []

        session.run.side_effect = run_side_effect
        engine = RetrievalEngine(driver=driver, embedder=None)
        result = engine._expand_entity_hits(["Test Entity"])
        types = {c["type"] for c in result}
        assert "entity" in types
        assert "join" in types
        assert "value" in types
        assert "metric" in types


class TestExpandFromEntitiesCharacterization:
    """Characterization tests capturing current behavior of expand_from_entities."""

    def test_expand_returns_physical_joins_and_values(self, mock_driver):
        driver, session = mock_driver

        def make_physical():
            return {
                "table_name": "cancer_diagnosis",
                "schema_name": "clinical",
                "catalog": "cdm",
                "columns": [
                    {"property": "Diagnosis Type", "column": "dx_type_cd",
                     "data_type": "STRING", "semantic_type": "categorical"},
                    {"property": "Stage", "column": "tnm_stage",
                     "data_type": "STRING", "semantic_type": "categorical"},
                ],
            }

        def make_join():
            return {
                "from_table": "cancer_diagnosis",
                "to_table": "cancer_surgery",
                "on_column": "patient_id",
                "cardinality": "one-to-many",
                "confidence": 0.85,
            }

        def make_value(code, label):
            return {"code": code, "label": label}

        def make_metric():
            return {
                "name": "survival_rate",
                "description": "5-year survival rate",
                "formula": "count(alive)/count(*)",
                "confidence": 0.7,
            }

        # session.run is called for: physical mapping, join paths,
        # value sets (one per categorical column), metrics, ancestry
        call_count = [0]
        physical_result = make_physical()

        def run_side_effect(query, **params):
            call_count[0] += 1
            r = MagicMock()
            # resolve_physical_mapping (new: ENTITY_ON_TABLE)
            if "ENTITY_ON_TABLE" in query and "entity_name" in params:
                r.data.return_value = physical_result
                return [r]
            # find_join_paths (new: JoinPath nodes)
            elif "JoinPath" in query or "USES" in query:
                r.data.return_value = make_join()
                return [r]
            # expand_value_set
            elif "MEMBER_OF" in query:
                r1 = MagicMock()
                r1.data.return_value = make_value("CRC", "Colorectal Cancer")
                r2 = MagicMock()
                r2.data.return_value = make_value("BRCA", "Breast Cancer")
                return [r1, r2]
            # expand_metrics
            elif "MEASURES" in query:
                r.data.return_value = make_metric()
                return [r]
            return []

        session.run.side_effect = run_side_effect

        engine = RetrievalEngine(driver=driver, embedder=None)
        result = engine.expand_from_entities(["Cancer Diagnosis"])

        # Returns a dict with expected keys
        assert isinstance(result, dict)
        assert "physical" in result
        assert "joins" in result
        assert "values" in result
        assert "ancestry" in result
        assert "metrics" in result

        # Physical has the table data
        assert len(result["physical"]) >= 1
        assert result["physical"][0]["table_name"] == "cancer_diagnosis"

        # Joins populated
        assert len(result["joins"]) >= 1
        assert result["joins"][0]["from_table"] == "cancer_diagnosis"

        # Values populated from categorical columns
        assert len(result["values"]) >= 1
        # Each value entry should have property, column, table keys added
        for v in result["values"]:
            assert "property" in v
            assert "column" in v
            assert "table" in v
