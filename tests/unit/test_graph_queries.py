import pytest

pytestmark = pytest.mark.unit

from sema.graph.queries import CypherQueries


class TestVectorSearch:
    def test_vector_search_query(self):
        q = CypherQueries.vector_search("entity_embeddings", 5)
        assert "db.index.vector.queryNodes" in q or "vector" in q.lower()
        assert "entity_embeddings" in q

    def test_vector_search_multiple_indexes(self):
        for index in ["entity_embeddings", "property_embeddings", "term_embeddings",
                       "synonym_embeddings", "metric_embeddings", "transformation_embeddings"]:
            q = CypherQueries.vector_search(index, 10)
            assert index in q


class TestAncestryTraversal:
    def test_parent_of_expansion(self):
        q = CypherQueries.expand_ancestry(max_depth=3)
        assert "PARENT_OF" in q
        assert "*1..3" in q or "1..3" in q

    def test_default_depth(self):
        q = CypherQueries.expand_ancestry()
        assert "PARENT_OF" in q


class TestValueSetExpansion:
    def test_member_of_expansion(self):
        q = CypherQueries.expand_value_set()
        assert "MEMBER_OF" in q


class TestSchemaMapping:
    def test_implemented_by_traversal(self):
        q = CypherQueries.resolve_physical_mapping()
        assert "IMPLEMENTED_BY" in q
        assert "Table" in q or "Column" in q


class TestJoinPathDiscovery:
    def test_candidate_join_query(self):
        q = CypherQueries.find_join_paths()
        assert "CANDIDATE_JOIN" in q


class TestMetricEntity:
    def test_measures_traversal(self):
        q = CypherQueries.expand_metrics()
        assert "MEASURES" in q


class TestTransformationLineage:
    def test_depends_on_produces(self):
        q = CypherQueries.expand_transformations()
        assert "DEPENDS_ON" in q
        assert "PRODUCES" in q


class TestAssertionQueries:
    def test_get_assertions_for_node(self):
        q = CypherQueries.get_assertions_for_subject()
        assert "SUBJECT" in q
        assert "Assertion" in q

    def test_get_all_assertions_by_run(self):
        q = CypherQueries.get_assertions_by_run()
        assert "run_id" in q
