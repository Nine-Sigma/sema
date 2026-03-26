import json
import pytest
from datetime import datetime, timezone

pytestmark = pytest.mark.integration

from sema.graph.loader import GraphLoader
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)


def _make_assertion(subject, predicate, payload=None, object_ref=None,
                    source="test", confidence=0.9, status=AssertionStatus.AUTO,
                    run_id="run-1"):
    return Assertion(
        id=f"a-{hash(subject + predicate.value + run_id) % 100000}",
        subject_ref=subject,
        predicate=predicate,
        payload=payload or {},
        object_ref=object_ref,
        source=source,
        confidence=confidence,
        status=status,
        run_id=run_id,
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def loader(clean_neo4j):
    return GraphLoader(clean_neo4j)


def _count(driver, label):
    with driver.session() as s:
        return s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]


def _get_node(driver, label, prop, value):
    with driver.session() as s:
        return s.run(f"MATCH (n:{label} {{{prop}: $v}}) RETURN n", v=value).single()


def _count_rels(driver, rel_type):
    with driver.session() as s:
        return s.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c").single()["c"]


class TestPhysicalNodeCreation:
    def test_upsert_catalog(self, loader, clean_neo4j):
        loader.upsert_catalog("cdm")
        assert _count(clean_neo4j, "Catalog") == 1

    def test_upsert_schema_with_catalog_edge(self, loader, clean_neo4j):
        loader.upsert_catalog("cdm")
        loader.upsert_schema("clinical", "cdm")
        assert _count(clean_neo4j, "Schema") == 1
        assert _count_rels(clean_neo4j, "IN_CATALOG") == 1

    def test_upsert_table_with_schema_edge(self, loader, clean_neo4j):
        loader.upsert_catalog("cdm")
        loader.upsert_schema("clinical", "cdm")
        loader.upsert_table("cancer_diagnosis", "clinical", "cdm",
                           table_type="TABLE", comment="Diagnosis records")
        assert _count(clean_neo4j, "Table") == 1
        assert _count_rels(clean_neo4j, "IN_SCHEMA") == 1
        node = _get_node(clean_neo4j, "Table", "name", "cancer_diagnosis")
        assert node["n"]["table_type"] == "TABLE"
        assert node["n"]["comment"] == "Diagnosis records"

    def test_upsert_column_with_table_edge(self, loader, clean_neo4j):
        loader.upsert_table("tbl", "schema", "catalog")
        loader.upsert_column("col1", "tbl", "schema", "catalog",
                            data_type="STRING", nullable=True, comment="Test col")
        assert _count(clean_neo4j, "Column") == 1
        assert _count_rels(clean_neo4j, "IN_TABLE") == 1
        node = _get_node(clean_neo4j, "Column", "name", "col1")
        assert node["n"]["data_type"] == "STRING"
        assert node["n"]["nullable"] is True


class TestUpsertIdempotency:
    def test_table_upsert_is_idempotent(self, loader, clean_neo4j):
        loader.upsert_table("tbl", "schema", "catalog", table_type="TABLE")
        loader.upsert_table("tbl", "schema", "catalog", table_type="TABLE")
        assert _count(clean_neo4j, "Table") == 1

    def test_column_upsert_updates_properties(self, loader, clean_neo4j):
        loader.upsert_table("tbl", "schema", "catalog")
        loader.upsert_column("col", "tbl", "schema", "catalog",
                            data_type="STRING", nullable=True, comment=None)
        loader.upsert_column("col", "tbl", "schema", "catalog",
                            data_type="INT", nullable=False, comment="Updated")
        assert _count(clean_neo4j, "Column") == 1
        node = _get_node(clean_neo4j, "Column", "name", "col")
        assert node["n"]["data_type"] == "INT"
        assert node["n"]["comment"] == "Updated"


class TestAssertionSupersession:
    def test_new_assertion_supersedes_old(self, loader, clean_neo4j):
        old = _make_assertion("ref://col", AssertionPredicate.HAS_LABEL,
                             {"value": "Old"}, run_id="run-1")
        loader.store_assertion(old)

        new = _make_assertion("ref://col", AssertionPredicate.HAS_LABEL,
                             {"value": "New"}, run_id="run-2")
        loader.store_assertion(new)

        with clean_neo4j.session() as s:
            results = list(s.run(
                "MATCH (a:Assertion) WHERE a.subject_ref = 'ref://col' "
                "RETURN a.status AS status, a.run_id AS run_id "
                "ORDER BY a.run_id"
            ))
        assert len(results) == 2
        assert results[0]["status"] == "superseded"
        assert results[0]["run_id"] == "run-1"
        assert results[1]["status"] == "auto"
        assert results[1]["run_id"] == "run-2"

    def test_pinned_assertion_not_superseded(self, loader, clean_neo4j):
        pinned = _make_assertion("ref://col", AssertionPredicate.HAS_LABEL,
                                {"value": "Pinned"}, run_id="run-1",
                                status=AssertionStatus.PINNED)
        loader.store_assertion(pinned)

        new = _make_assertion("ref://col", AssertionPredicate.HAS_LABEL,
                             {"value": "New"}, run_id="run-2")
        loader.store_assertion(new)

        with clean_neo4j.session() as s:
            results = list(s.run(
                "MATCH (a:Assertion) WHERE a.subject_ref = 'ref://col' "
                "RETURN a.status AS status, a.run_id AS run_id "
                "ORDER BY a.run_id"
            ))
        assert len(results) == 2
        # Pinned stays pinned
        assert results[0]["status"] == "pinned"
        assert results[1]["status"] == "auto"

    def test_different_sources_coexist(self, loader, clean_neo4j):
        a1 = _make_assertion("ref://col", AssertionPredicate.HAS_LABEL,
                            {"value": "Unity"}, source="unity_catalog", run_id="run-1")
        a2 = _make_assertion("ref://col", AssertionPredicate.HAS_LABEL,
                            {"value": "LLM"}, source="llm_interpretation", run_id="run-1")
        loader.store_assertion(a1)
        loader.store_assertion(a2)

        with clean_neo4j.session() as s:
            count = s.run(
                "MATCH (a:Assertion) WHERE a.subject_ref = 'ref://col' "
                "AND a.status = 'auto' RETURN count(a) AS c"
            ).single()["c"]
        assert count == 2

    def test_human_override_preserved_across_rebuild(self, loader, clean_neo4j):
        accepted = _make_assertion("ref://col", AssertionPredicate.HAS_LABEL,
                                  {"value": "Accepted"}, run_id="run-1",
                                  status=AssertionStatus.ACCEPTED)
        loader.store_assertion(accepted)

        new = _make_assertion("ref://col", AssertionPredicate.HAS_LABEL,
                             {"value": "New"}, run_id="run-2")
        loader.store_assertion(new)

        with clean_neo4j.session() as s:
            results = list(s.run(
                "MATCH (a:Assertion) WHERE a.subject_ref = 'ref://col' "
                "RETURN a.status AS status ORDER BY a.run_id"
            ))
        assert results[0]["status"] == "accepted"  # preserved


class TestCandidateJoins:
    def test_candidate_join_edge_created(self, loader, clean_neo4j):
        loader.upsert_table("tbl1", "schema", "catalog")
        loader.upsert_table("tbl2", "schema", "catalog")
        loader.upsert_candidate_join(
            "tbl1", "schema", "catalog",
            "tbl2", "schema", "catalog",
            on_column="patient_id", cardinality="one-to-many",
            source="heuristic", confidence=0.8,
        )
        assert _count_rels(clean_neo4j, "CANDIDATE_JOIN") == 1

        with clean_neo4j.session() as s:
            r = s.run(
                "MATCH ()-[j:CANDIDATE_JOIN]->() RETURN j.on_column AS on_col"
            ).single()
        assert r["on_col"] == "patient_id"
