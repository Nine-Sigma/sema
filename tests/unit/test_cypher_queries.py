"""Tests for CypherQueries static method contracts."""

import pytest

from sema.graph.queries import CypherQueries

pytestmark = pytest.mark.unit


class TestLexicalSearchQueries:
    def test_lexical_search_entities(self) -> None:
        q = CypherQueries.lexical_search_entities()
        assert "Entity" in q
        assert "toLower" in q
        assert "CONTAINS" in q
        assert "$token" in q

    def test_lexical_search_properties(self) -> None:
        q = CypherQueries.lexical_search_properties()
        assert "Property" in q
        assert "p.name" in q
        assert "CONTAINS" in q

    def test_lexical_search_terms(self) -> None:
        q = CypherQueries.lexical_search_terms()
        assert "Term" in q
        assert "t.label" in q
        assert "CONTAINS" in q

    def test_lexical_search_aliases(self) -> None:
        q = CypherQueries.lexical_search_aliases()
        assert "Alias" in q
        assert "a.text" in q
        assert "CONTAINS" in q

    def test_lexical_search_metrics(self) -> None:
        q = CypherQueries.lexical_search_metrics()
        assert "Metric" in q
        assert "m.name" in q
        assert "CONTAINS" in q


class TestLookupQueries:
    def test_find_entity_for_property(self) -> None:
        q = CypherQueries.find_entity_for_property()
        assert "HAS_PROPERTY" in q
        assert "$property_name" in q
        assert "entity_name" in q

    def test_find_column_for_property(self) -> None:
        q = CypherQueries.find_column_for_property()
        assert "PROPERTY_ON_COLUMN" in q
        assert "column_name" in q
        assert "table_name" in q

    def test_find_value_sets_for_term(self) -> None:
        q = CypherQueries.find_value_sets_for_term()
        assert "MEMBER_OF" in q
        assert "HAS_VALUE_SET" in q
        assert "$code" in q

    def test_find_vocabulary_for_term(self) -> None:
        q = CypherQueries.find_vocabulary_for_term()
        assert "IN_VOCABULARY" in q
        assert "$code" in q

    def test_dereference_alias(self) -> None:
        q = CypherQueries.dereference_alias()
        assert "REFERS_TO" in q
        assert "$text" in q
        assert "labels(target)" in q

    def test_find_property_vocabulary(self) -> None:
        q = CypherQueries.find_property_vocabulary()
        assert "CLASSIFIED_AS" in q
        assert "Vocabulary" in q


class TestAncestryDirection:
    def test_ancestry_traverses_upward(self) -> None:
        q = CypherQueries.expand_ancestry()
        assert "PARENT_OF" in q
        assert "ancestor" in q
        # Should traverse from child upward to ancestors
        assert "<-[:PARENT_OF" in q
