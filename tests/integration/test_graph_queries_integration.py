import pytest

pytestmark = pytest.mark.integration

from sema.graph.loader import GraphLoader
from sema.graph.queries import CypherQueries


@pytest.fixture
def loaded_graph(clean_neo4j):
    """Load a small test graph for query testing."""
    loader = GraphLoader(clean_neo4j)

    # Physical layer
    loader.upsert_catalog("cdm")
    loader.upsert_schema("clinical", "cdm")
    loader.upsert_table("cancer_diagnosis", "clinical", "cdm", table_type="TABLE")
    loader.upsert_table("cancer_surgery", "clinical", "cdm", table_type="TABLE")
    loader.upsert_column("dx_type_cd", "cancer_diagnosis", "clinical", "cdm",
                        data_type="STRING", nullable=True)
    loader.upsert_column("patient_id", "cancer_diagnosis", "clinical", "cdm",
                        data_type="STRING", nullable=False)

    # Semantic layer
    loader.upsert_entity("Cancer Diagnosis", description="Primary dx",
                        source="llm", confidence=0.8,
                        table_name="cancer_diagnosis", schema_name="clinical", catalog="cdm")
    loader.upsert_property("Diagnosis Type", semantic_type="categorical",
                          source="llm", confidence=0.8,
                          entity_name="Cancer Diagnosis",
                          column_name="dx_type_cd", table_name="cancer_diagnosis",
                          schema_name="clinical", catalog="cdm")

    # Terms and hierarchy
    loader.upsert_term("CRC", "Colorectal Cancer", source="llm", confidence=0.85)
    loader.upsert_term("COAD", "Colon Adenocarcinoma", source="llm", confidence=0.85)
    loader.upsert_term("READ", "Rectal Adenocarcinoma", source="llm", confidence=0.85)
    loader.upsert_value_set("dx_types", column_name="dx_type_cd",
                           table_name="cancer_diagnosis", schema_name="clinical", catalog="cdm")
    loader.add_term_to_value_set("CRC", "dx_types")
    loader.add_term_to_value_set("COAD", "dx_types")
    loader.add_term_hierarchy("CRC", "COAD")
    loader.add_term_hierarchy("CRC", "READ")

    # Synonym
    loader.upsert_synonym("colon cancer", parent_label=":Entity",
                         parent_name="Cancer Diagnosis", source="llm", confidence=0.8)

    # Join
    loader.upsert_candidate_join(
        "cancer_diagnosis", "clinical", "cdm",
        "cancer_surgery", "clinical", "cdm",
        on_column="patient_id", cardinality="one-to-many",
        source="heuristic", confidence=0.8,
    )

    return clean_neo4j


class TestAncestryTraversal:
    def test_expand_children(self, loaded_graph):
        query = CypherQueries.expand_ancestry(max_depth=3)
        with loaded_graph.session() as s:
            results = list(s.run(query, code="CRC"))
        codes = {r["code"] for r in results}
        assert "COAD" in codes
        assert "READ" in codes


class TestValueSetExpansion:
    def test_expand_value_set_members(self, loaded_graph):
        query = CypherQueries.expand_value_set()
        with loaded_graph.session() as s:
            results = list(s.run(query, value_set_name="dx_types"))
        codes = {r["code"] for r in results}
        assert "CRC" in codes
        assert "COAD" in codes


class TestPhysicalMapping:
    def test_resolve_entity_to_table(self, loaded_graph):
        query = CypherQueries.resolve_physical_mapping()
        with loaded_graph.session() as s:
            results = list(s.run(query, entity_name="Cancer Diagnosis"))
        assert len(results) == 1
        assert results[0]["table_name"] == "cancer_diagnosis"
        columns = results[0]["columns"]
        col_names = [c["column"] for c in columns]
        assert "dx_type_cd" in col_names


class TestJoinPaths:
    def test_find_joins_for_tables(self, loaded_graph):
        query = CypherQueries.find_join_paths()
        with loaded_graph.session() as s:
            results = list(s.run(query, table_names=["cancer_diagnosis", "cancer_surgery"]))
        assert len(results) >= 1
        assert results[0]["on_column"] == "patient_id"


class TestAssertionQueries:
    def test_get_assertions_for_subject(self, loaded_graph):
        # Store an assertion first
        loader = GraphLoader(loaded_graph)
        from sema.models.assertions import Assertion, AssertionPredicate
        from datetime import datetime, timezone
        a = Assertion(
            id="test-assertion",
            subject_ref="unity://cdm.clinical.cancer_diagnosis",
            predicate=AssertionPredicate.HAS_LABEL,
            payload={"value": "Cancer Diagnosis Table"},
            source="test",
            confidence=0.9,
            run_id="run-test",
            observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        loader.store_assertion(a)

        # Link assertion to the table node via SUBJECT edge
        with loaded_graph.session() as s:
            s.run(
                "MATCH (a:Assertion {id: 'test-assertion'}), "
                "(t:Table {name: 'cancer_diagnosis'}) "
                "MERGE (a)-[:SUBJECT]->(t)"
            )

        query = CypherQueries.get_assertions_for_subject()
        with loaded_graph.session() as s:
            results = list(s.run(query, name="cancer_diagnosis"))
        assert len(results) >= 1
        assert results[0]["source"] == "test"
