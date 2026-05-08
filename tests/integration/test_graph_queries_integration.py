import pytest

pytestmark = pytest.mark.integration

from sema.graph.loader import GraphLoader
from sema.graph.queries import CypherQueries


@pytest.fixture
def loaded_graph(clean_neo4j):
    """Load a small test graph for query testing."""
    loader = GraphLoader(clean_neo4j)

    # Physical layer (refs let `add_join_path_uses` MATCH the tables)
    loader.upsert_catalog("cdm")
    loader.upsert_schema("clinical", "cdm")
    diag_ref = "unity://cdm.clinical.cancer_diagnosis"
    surg_ref = "unity://cdm.clinical.cancer_surgery"
    loader.upsert_table("cancer_diagnosis", "clinical", "cdm",
                        table_type="TABLE", ref=diag_ref)
    loader.upsert_table("cancer_surgery", "clinical", "cdm",
                        table_type="TABLE", ref=surg_ref)
    loader.upsert_column("dx_type_cd", "cancer_diagnosis", "clinical", "cdm",
                        data_type="STRING", nullable=True)
    loader.upsert_column("patient_id", "cancer_diagnosis", "clinical", "cdm",
                        data_type="STRING", nullable=False)

    # Semantic layer
    loader.upsert_entity("Cancer Diagnosis", description="Primary dx",
                        source="llm", confidence=0.8,
                        table_name="cancer_diagnosis", schema_name="clinical", catalog="cdm",
                        source_schema="clinical")
    loader.upsert_property("Diagnosis Type", semantic_type="categorical",
                          source="llm", confidence=0.8,
                          entity_name="Cancer Diagnosis",
                          column_name="dx_type_cd", table_name="cancer_diagnosis",
                          schema_name="clinical", catalog="cdm",
                          source_schema="clinical")

    # Terms and hierarchy
    loader.upsert_term("CRC", "Colorectal Cancer", source="llm", confidence=0.85)
    loader.upsert_term("COAD", "Colon Adenocarcinoma", source="llm", confidence=0.85)
    loader.upsert_term("READ", "Rectal Adenocarcinoma", source="llm", confidence=0.85)
    loader.upsert_value_set("dx_types", column_name="dx_type_cd",
                           table_name="cancer_diagnosis", schema_name="clinical", catalog="cdm",
                           source_schema="clinical")
    loader.add_term_to_value_set("CRC", "dx_types", source_schema="clinical")
    loader.add_term_to_value_set("COAD", "dx_types", source_schema="clinical")
    loader.add_term_hierarchy("CRC", "COAD", source_schema="clinical")
    loader.add_term_hierarchy("CRC", "READ", source_schema="clinical")

    # Alias
    loader.upsert_alias("colon cancer", parent_label=":Entity",
                        parent_name="Cancer Diagnosis", source="llm", confidence=0.8,
                        source_schema="clinical")

    # Join — production wiring: the JoinPath node + :USES edges to each
    # contributing Table (the CypherQueries.find_join_paths query MATCHes
    # `(jp:JoinPath)-[:USES]->(t:Table)`).
    jp_name = "cancer_diagnosis__cancer_surgery__patient_id"
    loader.upsert_join_path(
        name=jp_name,
        join_predicates=[{"from_table": "cancer_diagnosis", "to_table": "cancer_surgery", "on_column": "patient_id"}],
        hop_count=1,
        source="heuristic", confidence=0.8,
        source_schema="clinical",
    )
    loader.add_join_path_uses(jp_name, diag_ref, source_schema="clinical")
    loader.add_join_path_uses(jp_name, surg_ref, source_schema="clinical")

    return clean_neo4j


class TestAncestryTraversal:
    def test_expand_ancestors_of_descendant(self, loaded_graph):
        # Fixture wires CRC -[:PARENT_OF]-> COAD and CRC -[:PARENT_OF]-> READ.
        # `expand_ancestry` traverses INCOMING PARENT_OF (upward to parents),
        # so COAD's ancestor is CRC.
        query = CypherQueries.expand_ancestry(max_depth=3)
        with loaded_graph.session() as s:
            results = list(s.run(query, code="COAD", vocabulary_name=None))
        codes = {r["code"] for r in results}
        assert "CRC" in codes


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
        import json as _json
        query = CypherQueries.find_join_paths()
        with loaded_graph.session() as s:
            results = list(s.run(query, table_names=["cancer_diagnosis", "cancer_surgery"]))
        assert len(results) >= 1
        jp = results[0]
        assert jp["name"] == "cancer_diagnosis__cancer_surgery__patient_id"
        predicates = _json.loads(jp["join_predicates"])
        assert predicates[0]["on_column"] == "patient_id"


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
