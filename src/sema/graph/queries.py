from __future__ import annotations


class CypherQueries:
    """Parameterized Cypher query templates for the semantic graph."""

    @staticmethod
    def vector_search(index_name: str, top_k: int = 10) -> str:
        return (
            f"CALL db.index.vector.queryNodes('{index_name}', {top_k}, $embedding) "
            f"YIELD node, score "
            f"RETURN node, score ORDER BY score DESC"
        )

    @staticmethod
    def expand_ancestry(max_depth: int = 3) -> str:
        return (
            f"MATCH (t:Term)<-[:PARENT_OF*1..{max_depth}]-(ancestor:Term) "
            f"WHERE t.code = $code "
            f"AND (t.vocabulary_name = $vocabulary_name "
            f"OR $vocabulary_name IS NULL) "
            f"RETURN ancestor.code AS code, ancestor.label AS label, "
            f"ancestor.vocabulary_name AS vocabulary_name, "
            f"t.code AS parent_code"
        )

    @staticmethod
    def expand_value_set() -> str:
        return (
            "MATCH (t:Term)-[:MEMBER_OF]->(vs:ValueSet {name: $value_set_name}) "
            "RETURN t.code AS code, t.label AS label"
        )

    @staticmethod
    def resolve_physical_mapping() -> str:
        return (
            "MATCH (e:Entity {name: $entity_name})-[:ENTITY_ON_TABLE]->(t:Table) "
            "OPTIONAL MATCH (e)-[:HAS_PROPERTY]->(p:Property)"
            "-[:PROPERTY_ON_COLUMN]->(c:Column) "
            "RETURN t.name AS table_name, t.schema_name AS schema_name, "
            "t.catalog AS catalog, "
            "collect({property: p.name, column: c.name, "
            "data_type: c.data_type, semantic_type: p.semantic_type}) AS columns"
        )

    @staticmethod
    def find_join_paths() -> str:
        return (
            "MATCH (jp:JoinPath)-[:USES]->(t:Table) "
            "WHERE t.name IN $table_names "
            "RETURN jp.id AS id, jp.name AS name, "
            "jp.join_predicates AS join_predicates, "
            "jp.hop_count AS hop_count, "
            "jp.cardinality_hint AS cardinality_hint, "
            "jp.sql_snippet AS sql_snippet, "
            "jp.confidence AS confidence"
        )

    @staticmethod
    def expand_metrics() -> str:
        return (
            "MATCH (m:Metric)-[:MEASURES]->(e:Entity {name: $entity_name}) "
            "OPTIONAL MATCH (m)-[:AGGREGATES]->(p:Property) "
            "OPTIONAL MATCH (m)-[:FILTERS_BY]->(f) "
            "OPTIONAL MATCH (m)-[:AT_GRAIN]->(g) "
            "RETURN m.name AS name, m.description AS description, "
            "m.formula AS formula, m.grain AS grain, "
            "m.confidence AS confidence, "
            "collect(DISTINCT p.name) AS aggregates, "
            "collect(DISTINCT f.name) AS filters, "
            "collect(DISTINCT g.name) AS grains"
        )

    @staticmethod
    def expand_aliases() -> str:
        return (
            "MATCH (a:Alias)-[:REFERS_TO]->(n) "
            "WHERE n.name = $name "
            "RETURN a.text AS text, a.description AS description, "
            "a.is_preferred AS is_preferred, "
            "a.confidence AS confidence"
        )

    @staticmethod
    def get_provenance() -> str:
        return (
            "MATCH (a:Assertion)-[:SUBJECT]->(n) "
            "WHERE n.id = $node_id "
            "OPTIONAL MATCH (a)-[:OBJECT]->(o) "
            "RETURN a.id AS assertion_id, a.predicate AS predicate, "
            "a.payload AS payload, a.source AS source, "
            "a.confidence AS confidence, a.status AS status, "
            "o.id AS object_id"
        )

    @staticmethod
    def get_assertions_for_subject() -> str:
        return (
            "MATCH (a:Assertion)-[:SUBJECT]->(n) "
            "WHERE n.name = $name "
            "RETURN a.predicate AS predicate, a.payload AS payload, "
            "a.source AS source, a.confidence AS confidence, "
            "a.status AS status, a.run_id AS run_id"
        )

    # --- Lexical search queries ---

    @staticmethod
    def lexical_search_entities() -> str:
        return (
            "MATCH (e:Entity) "
            "WHERE toLower(e.name) CONTAINS $token "
            "RETURN e AS node"
        )

    @staticmethod
    def lexical_search_properties() -> str:
        return (
            "MATCH (p:Property) "
            "WHERE toLower(p.name) CONTAINS $token "
            "RETURN p AS node"
        )

    @staticmethod
    def lexical_search_terms() -> str:
        return (
            "MATCH (t:Term) "
            "WHERE toLower(t.label) CONTAINS $token "
            "RETURN t AS node"
        )

    @staticmethod
    def lexical_search_aliases() -> str:
        return (
            "MATCH (a:Alias) "
            "WHERE toLower(a.text) CONTAINS $token "
            "RETURN a AS node"
        )

    @staticmethod
    def lexical_search_metrics() -> str:
        return (
            "MATCH (m:Metric) "
            "WHERE toLower(m.name) CONTAINS $token "
            "RETURN m AS node"
        )

    # --- Lookup helpers ---

    @staticmethod
    def find_entity_for_property() -> str:
        return (
            "MATCH (e:Entity)-[:HAS_PROPERTY]->(p:Property) "
            "WHERE p.name = $property_name "
            "AND (p.entity_name = $entity_name "
            "OR $entity_name IS NULL) "
            "RETURN e.name AS entity_name, "
            "e.description AS description"
        )

    @staticmethod
    def find_column_for_property() -> str:
        return (
            "MATCH (p:Property)-[:PROPERTY_ON_COLUMN]->(c:Column) "
            "WHERE p.name = $property_name "
            "AND (p.entity_name = $entity_name "
            "OR $entity_name IS NULL) "
            "RETURN c.name AS column_name, "
            "c.table_name AS table_name, "
            "c.schema_name AS schema_name, "
            "c.catalog AS catalog"
        )

    @staticmethod
    def find_value_sets_for_term() -> str:
        return (
            "MATCH (t:Term)-[:MEMBER_OF]->(vs:ValueSet)"
            "<-[:HAS_VALUE_SET]-(c:Column) "
            "WHERE t.code = $code "
            "OPTIONAL MATCH (c)-[:IN_TABLE]->(tbl:Table) "
            "RETURN vs.name AS value_set_name, "
            "c.name AS column_name, "
            "tbl.name AS table_name, "
            "tbl.schema_name AS schema_name, "
            "tbl.catalog AS catalog"
        )

    @staticmethod
    def find_vocabulary_for_term() -> str:
        return (
            "MATCH (t:Term {code: $code})"
            "-[:IN_VOCABULARY]->(v:Vocabulary) "
            "RETURN v.name AS vocabulary_name LIMIT 1"
        )

    @staticmethod
    def dereference_alias() -> str:
        return (
            "MATCH (a:Alias)-[:REFERS_TO]->(target) "
            "WHERE a.text = $text "
            "RETURN target.name AS target_name, "
            "labels(target) AS target_labels, "
            "target.entity_name AS entity_name"
        )

    @staticmethod
    def find_value_set_members_by_column() -> str:
        return (
            "MATCH (c:Column {name: $column_name, "
            "table_name: $table_name})"
            "-[:HAS_VALUE_SET]->(vs:ValueSet) "
            "MATCH (t:Term)-[:MEMBER_OF]->(vs) "
            "RETURN t.code AS code, t.label AS label"
        )

    @staticmethod
    def find_property_vocabulary() -> str:
        return (
            "MATCH (p:Property {datasource_id: $ds, "
            "column_key: $ck})"
            "-[:CLASSIFIED_AS]->(v:Vocabulary) "
            "RETURN v.name AS vocabulary_name"
        )

    @staticmethod
    def get_assertions_by_run() -> str:
        return (
            "MATCH (a:Assertion) "
            "WHERE a.run_id = $run_id "
            "RETURN a.id AS id, a.subject_ref AS subject_ref, "
            "a.predicate AS predicate, a.source AS source, "
            "a.confidence AS confidence, a.status AS status"
        )
