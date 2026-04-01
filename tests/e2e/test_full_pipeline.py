import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

pytestmark = pytest.mark.e2e

from sema.engine.structural import StructuralEngine
from sema.engine.semantic import SemanticEngine
from sema.engine.vocabulary import VocabularyEngine
from sema.engine.embeddings import EmbeddingEngine, build_embedding_text
from sema.graph.loader import GraphLoader
from sema.graph.materializer import materialize_unified
from sema.pipeline.context import prune_to_sco
from sema.consumers.nl2sql.consumer import NL2SQLConsumer
from sema.consumers.nl2sql.prompting import build_sql_prompt
from sema.consumers.nl2sql.validation import validate_sql_against_sco
from sema.models.assertions import (
    Assertion, AssertionPredicate, AssertionStatus,
)
from sema.models.context import SemanticCandidateSet

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _make_extraction_assertions():
    """Simulate Databricks connector output for 2 tables."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    run_id = "e2e-run-1"
    assertions = []

    def _a(subject, predicate, payload=None, object_ref=None, conf=1.0):
        assertions.append(Assertion(
            id=f"ext-{len(assertions)}",
            subject_ref=subject, predicate=predicate,
            payload=payload or {}, object_ref=object_ref,
            source="unity_catalog", confidence=conf,
            run_id=run_id, observed_at=now,
        ))

    # Table: cancer_diagnosis
    _a("unity://cdm.clinical.cancer_diagnosis",
       AssertionPredicate.TABLE_EXISTS, {"table_type": "TABLE"})
    _a("unity://cdm.clinical.cancer_diagnosis.dx_type_cd",
       AssertionPredicate.COLUMN_EXISTS,
       {"data_type": "STRING", "nullable": True, "comment": "Diagnosis type"})
    _a("unity://cdm.clinical.cancer_diagnosis.tnm_stage",
       AssertionPredicate.COLUMN_EXISTS,
       {"data_type": "STRING", "nullable": True, "comment": None})
    _a("unity://cdm.clinical.cancer_diagnosis.patient_id",
       AssertionPredicate.COLUMN_EXISTS,
       {"data_type": "STRING", "nullable": False, "comment": "Patient ID"})
    _a("unity://cdm.clinical.cancer_diagnosis.date_of_diagnosis",
       AssertionPredicate.COLUMN_EXISTS,
       {"data_type": "DATE", "nullable": True, "comment": None})
    _a("unity://cdm.clinical.cancer_diagnosis.dx_type_cd",
       AssertionPredicate.HAS_TOP_VALUES,
       {"values": [{"value": "CRC", "frequency": 100}, {"value": "BRCA", "frequency": 80}],
        "approx_distinct": 10})
    _a("unity://cdm.clinical.cancer_diagnosis.tnm_stage",
       AssertionPredicate.HAS_TOP_VALUES,
       {"values": [{"value": "Stage I", "frequency": 50}, {"value": "Stage III", "frequency": 40},
                   {"value": "Stage IIIA", "frequency": 20}],
        "approx_distinct": 8})
    _a("unity://cdm.clinical.cancer_diagnosis",
       AssertionPredicate.HAS_SAMPLE_ROWS,
       {"rows": [{"dx_type_cd": "CRC", "tnm_stage": "Stage III", "patient_id": "P1", "date_of_diagnosis": "2024-01-15"}],
        "columns": ["dx_type_cd", "tnm_stage", "patient_id", "date_of_diagnosis"]})

    # Table: cancer_surgery
    _a("unity://cdm.clinical.cancer_surgery",
       AssertionPredicate.TABLE_EXISTS, {"table_type": "TABLE"})
    _a("unity://cdm.clinical.cancer_surgery.patient_id",
       AssertionPredicate.COLUMN_EXISTS,
       {"data_type": "STRING", "nullable": False, "comment": None})
    _a("unity://cdm.clinical.cancer_surgery.procedure_date",
       AssertionPredicate.COLUMN_EXISTS,
       {"data_type": "DATE", "nullable": True, "comment": None})

    # FK join
    _a("unity://cdm.clinical.cancer_diagnosis",
       AssertionPredicate.JOINS_TO,
       {"on_column": "patient_id", "cardinality": "one-to-many"},
       object_ref="unity://cdm.clinical.cancer_surgery")

    return assertions


def _mock_llm_response():
    with open(FIXTURES / "expected_llm_response.json") as f:
        return json.load(f)


class FakeEmbedder:
    def encode(self, texts):
        return [[0.1] * 384 for _ in texts]


@pytest.fixture
def built_graph(clean_neo4j):
    """Build a complete graph from fixture data."""
    loader = GraphLoader(clean_neo4j)
    extraction_assertions = _make_extraction_assertions()

    # L1: Structural
    structural = StructuralEngine(loader)
    structural.process(extraction_assertions)

    # L2: Semantic (mocked LLM)
    llm_response = _mock_llm_response()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=json.dumps(llm_response))
    semantic = SemanticEngine(llm=mock_llm, run_id="e2e-run-1")

    table_meta = {
        "table_ref": "unity://cdm.clinical.cancer_diagnosis",
        "table_name": "cancer_diagnosis",
        "schema_name": "clinical",
        "catalog": "cdm",
        "comment": None,
        "columns": [
            {"name": "dx_type_cd", "data_type": "STRING", "nullable": True, "comment": "Diagnosis type",
             "top_values": [{"value": "CRC", "frequency": 100}]},
            {"name": "tnm_stage", "data_type": "STRING", "nullable": True, "comment": None,
             "top_values": [{"value": "Stage III", "frequency": 40}]},
            {"name": "patient_id", "data_type": "STRING", "nullable": False, "comment": None, "top_values": None},
            {"name": "date_of_diagnosis", "data_type": "DATE", "nullable": True, "comment": None, "top_values": None},
        ],
        "sample_rows": [{"dx_type_cd": "CRC", "tnm_stage": "Stage III"}],
    }
    semantic_assertions = semantic.interpret_table(table_meta)

    # L3: Vocabulary (mocked)
    vocab_llm = MagicMock()
    vocab_llm.invoke.return_value = MagicMock(content=json.dumps(
        {"synonyms": [{"term": "Colorectal Cancer", "synonyms": ["colon cancer"]}]}
    ))
    vocab = VocabularyEngine(llm=vocab_llm, run_id="e2e-run-1")
    vocab_assertions = vocab.process_column(
        "unity://cdm.clinical.cancer_diagnosis.tnm_stage",
        ["Stage I", "Stage III", "Stage IIIA"], None,
    )

    # Materialization
    materialize_unified(loader, semantic_assertions + vocab_assertions)

    # Embeddings
    emb_engine = EmbeddingEngine(model=FakeEmbedder(), loader=loader)
    emb_engine.create_all_indexes(dimensions=384)

    return clean_neo4j


def _count(driver, label):
    with driver.session() as s:
        return s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]


class TestE2EBuild:
    def test_graph_has_correct_nodes(self, built_graph):
        assert _count(built_graph, "Catalog") == 1
        assert _count(built_graph, "Schema") == 1
        assert _count(built_graph, "Table") == 2
        assert _count(built_graph, "Column") >= 4
        assert _count(built_graph, "Entity") >= 1
        assert _count(built_graph, "Property") >= 1
        assert _count(built_graph, "Term") >= 1
        assert _count(built_graph, "Assertion") >= 1

    def test_graph_has_correct_edges(self, built_graph):
        with built_graph.session() as s:
            for rel in ["IN_CATALOG", "IN_SCHEMA", "IN_TABLE",
                       "ENTITY_ON_TABLE", "HAS_PROPERTY"]:
                count = s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"]
                assert count >= 1, f"Missing {rel} edges"
            # JoinPath is a node, not an edge, in the v1 model
            jp_count = s.run("MATCH (jp:JoinPath) RETURN count(jp) AS c").single()["c"]
            assert jp_count >= 1, "Missing JoinPath nodes"


class TestE2EContext:
    def test_sco_from_candidates(self, built_graph):
        candidates = SemanticCandidateSet(
            query="stage 3 colorectal",
            candidates=[
                {"type": "entity", "name": "Cancer Diagnosis",
                 "table": "cancer_diagnosis", "schema": "clinical", "catalog": "cdm",
                 "description": "Primary dx", "confidence": 0.8, "source": "llm",
                 "columns": [
                     {"property": "Diagnosis Type", "column": "dx_type_cd",
                      "data_type": "STRING", "semantic_type": "categorical"},
                     {"property": "TNM Stage", "column": "tnm_stage",
                      "data_type": "STRING", "semantic_type": "categorical"},
                 ]},
                {"type": "join",
                 "from_table": "cancer_diagnosis", "from_schema": "clinical", "from_catalog": "cdm",
                 "to_table": "cancer_surgery", "to_schema": "clinical", "to_catalog": "cdm",
                 "on_column": "patient_id", "cardinality": "one-to-many", "confidence": 0.8},
                {"type": "value", "property_name": "TNM Stage",
                 "column": "tnm_stage", "table": "cancer_diagnosis",
                 "code": "Stage III", "label": "Stage III"},
            ],
        )
        sco = prune_to_sco(candidates, consumer="nl2sql")
        assert len(sco.entities) == 1
        assert len(sco.physical_assets) >= 1
        assert len(sco.join_paths) == 1
        assert len(sco.governed_values) >= 1


class TestE2EQuery:
    def test_generated_sql_is_valid(self, built_graph):
        candidates = SemanticCandidateSet(
            query="stage 3 patients",
            candidates=[
                {"type": "entity", "name": "Cancer Diagnosis",
                 "table": "cancer_diagnosis", "schema": "clinical", "catalog": "cdm",
                 "description": "Primary dx", "confidence": 0.8, "source": "llm",
                 "columns": [
                     {"property": "TNM Stage", "column": "tnm_stage",
                      "data_type": "STRING", "semantic_type": "categorical"},
                     {"property": "Patient ID", "column": "patient_id",
                      "data_type": "STRING", "semantic_type": "identifier"},
                 ]},
            ],
        )
        sco = prune_to_sco(candidates, consumer="nl2sql")

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="SELECT patient_id, tnm_stage FROM cdm.clinical.cancer_diagnosis WHERE tnm_stage = 'Stage III'"
        )
        from sema.consumers.base import ConsumerDeps, ConsumerRequest

        consumer = NL2SQLConsumer()
        deps = ConsumerDeps(llm=mock_llm)
        req = ConsumerRequest(
            question="stage 3 patients", operation="plan",
        )
        plan = consumer.plan(req, sco, deps)

        assert plan.valid
        assert "cancer_diagnosis" in plan.sql
        assert "tnm_stage" in plan.sql

        errors = validate_sql_against_sco(plan.sql, sco)
        assert len(errors) == 0


class TestE2ERebuild:
    def test_rebuild_preserves_human_overrides(self, clean_neo4j):
        loader = GraphLoader(clean_neo4j)

        # First build
        assertions_v1 = [
            Assertion(id="v1-entity", subject_ref="unity://cdm.clinical.tbl",
                     predicate=AssertionPredicate.HAS_ENTITY_NAME,
                     payload={"value": "Original Name"},
                     source="llm_interpretation", confidence=0.7,
                     status=AssertionStatus.AUTO, run_id="run-1",
                     observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),
        ]
        materialize_unified(loader, assertions_v1)

        # Human pins a different name
        pinned = Assertion(
            id="pinned-entity", subject_ref="unity://cdm.clinical.tbl",
            predicate=AssertionPredicate.HAS_ENTITY_NAME,
            payload={"value": "Human Curated Name"},
            source="human", confidence=1.0,
            status=AssertionStatus.PINNED, run_id="manual",
            observed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
        loader.store_assertion(pinned)

        # Second build
        assertions_v2 = [
            Assertion(id="v2-entity", subject_ref="unity://cdm.clinical.tbl",
                     predicate=AssertionPredicate.HAS_ENTITY_NAME,
                     payload={"value": "New LLM Name"},
                     source="llm_interpretation", confidence=0.8,
                     status=AssertionStatus.AUTO, run_id="run-2",
                     observed_at=datetime(2026, 1, 3, tzinfo=timezone.utc)),
        ]
        materialize_unified(loader, assertions_v2)

        # Pinned assertion should survive
        with clean_neo4j.session() as s:
            results = list(s.run(
                "MATCH (a:Assertion) WHERE a.subject_ref = 'unity://cdm.clinical.tbl' "
                "AND a.predicate = 'has_entity_name' "
                "RETURN a.status AS status, a.payload AS payload "
                "ORDER BY a.observed_at"
            ))
        statuses = [r["status"] for r in results]
        assert "pinned" in statuses
