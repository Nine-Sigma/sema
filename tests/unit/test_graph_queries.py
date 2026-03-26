import pytest

pytestmark = pytest.mark.unit

from sema.graph.queries import CypherQueries


class TestVectorSearch:
    def test_vector_search_query(self):
        q = CypherQueries.vector_search("entity_embedding_index", 5)
        assert "db.index.vector.queryNodes" in q or "vector" in q.lower()
        assert "entity_embedding_index" in q

    def test_vector_search_multiple_indexes(self):
        for index in [
            "entity_embedding_index",
            "property_embedding_index",
            "term_embedding_index",
            "alias_embedding_index",
            "metric_embedding_index",
        ]:
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

    def test_returns_parent_code(self):
        q = CypherQueries.expand_ancestry()
        assert "parent" in q.lower() or "ancestor" in q.lower()


class TestValueSetExpansion:
    def test_member_of_expansion(self):
        q = CypherQueries.expand_value_set()
        assert "MEMBER_OF" in q


class TestSchemaMapping:
    def test_entity_on_table_traversal(self):
        q = CypherQueries.resolve_physical_mapping()
        assert "ENTITY_ON_TABLE" in q
        assert "IMPLEMENTED_BY" not in q

    def test_property_on_column_traversal(self):
        q = CypherQueries.resolve_physical_mapping()
        assert "PROPERTY_ON_COLUMN" in q


class TestJoinPathDiscovery:
    def test_join_path_node_query(self):
        q = CypherQueries.find_join_paths()
        assert "JoinPath" in q
        assert "CANDIDATE_JOIN" not in q

    def test_returns_join_predicates(self):
        q = CypherQueries.find_join_paths()
        assert "join_predicates" in q

    def test_returns_hop_count(self):
        q = CypherQueries.find_join_paths()
        assert "hop_count" in q

    def test_returns_cardinality_hint(self):
        q = CypherQueries.find_join_paths()
        assert "cardinality_hint" in q

    def test_returns_sql_snippet(self):
        q = CypherQueries.find_join_paths()
        assert "sql_snippet" in q


class TestMetricExpansion:
    def test_measures_traversal(self):
        q = CypherQueries.expand_metrics()
        assert "MEASURES" in q

    def test_aggregates_traversal(self):
        q = CypherQueries.expand_metrics()
        assert "AGGREGATES" in q

    def test_filters_by_traversal(self):
        q = CypherQueries.expand_metrics()
        assert "FILTERS_BY" in q

    def test_at_grain_traversal(self):
        q = CypherQueries.expand_metrics()
        assert "AT_GRAIN" in q


class TestAliasQueries:
    def test_alias_refers_to(self):
        q = CypherQueries.expand_aliases()
        assert "REFERS_TO" in q
        assert "Alias" in q
        assert "SYNONYM_OF" not in q


class TestProvenanceQueries:
    def test_provenance_subject_object(self):
        q = CypherQueries.get_provenance()
        assert "SUBJECT" in q
        assert "OBJECT" in q
        assert "Assertion" in q


class TestAssertionQueries:
    def test_get_assertions_for_node(self):
        q = CypherQueries.get_assertions_for_subject()
        assert "SUBJECT" in q
        assert "Assertion" in q

    def test_get_all_assertions_by_run(self):
        q = CypherQueries.get_assertions_by_run()
        assert "run_id" in q


class TestNoDeadCode:
    def test_no_depends_on_produces(self):
        """Transformation queries removed per v1 model."""
        assert not hasattr(CypherQueries, "expand_transformations")
