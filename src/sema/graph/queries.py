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
            f"MATCH (t:Term)-[:PARENT_OF*1..{max_depth}]->(child:Term) "
            f"WHERE t.code = $code "
            f"RETURN child.code AS code, child.label AS label"
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
            "MATCH (e:Entity {name: $entity_name})-[:IMPLEMENTED_BY]->(t:Table) "
            "OPTIONAL MATCH (e)-[:HAS_PROPERTY]->(p:Property)-[:IMPLEMENTED_BY]->(c:Column) "
            "RETURN t.name AS table_name, t.schema_name AS schema_name, "
            "t.catalog AS catalog, "
            "collect({property: p.name, column: c.name, "
            "data_type: c.data_type, semantic_type: p.semantic_type}) AS columns"
        )

    @staticmethod
    def find_join_paths() -> str:
        return (
            "MATCH (t1:Table)-[j:CANDIDATE_JOIN]->(t2:Table) "
            "WHERE t1.name IN $table_names OR t2.name IN $table_names "
            "RETURN t1.name AS from_table, t1.schema_name AS from_schema, "
            "t1.catalog AS from_catalog, "
            "t2.name AS to_table, t2.schema_name AS to_schema, "
            "t2.catalog AS to_catalog, "
            "j.on_column AS on_column, j.cardinality AS cardinality, "
            "j.confidence AS confidence, j.semantic_label AS semantic_label"
        )

    @staticmethod
    def expand_metrics() -> str:
        return (
            "MATCH (m:Metric)-[:MEASURES]->(e:Entity {name: $entity_name}) "
            "RETURN m.name AS name, m.description AS description, "
            "m.formula AS formula, m.confidence AS confidence"
        )

    @staticmethod
    def expand_transformations() -> str:
        return (
            "MATCH (tr:Transformation)-[:DEPENDS_ON]->(src:Table) "
            "MATCH (tr)-[:PRODUCES]->(tgt:Table) "
            "WHERE src.name IN $table_names OR tgt.name IN $table_names "
            "RETURN tr.name AS name, tr.transform_type AS type, "
            "src.name AS depends_on, tgt.name AS produces, "
            "tr.confidence AS confidence"
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

    @staticmethod
    def get_assertions_by_run() -> str:
        return (
            "MATCH (a:Assertion) "
            "WHERE a.run_id = $run_id "
            "RETURN a.id AS id, a.subject_ref AS subject_ref, "
            "a.predicate AS predicate, a.source AS source, "
            "a.confidence AS confidence, a.status AS status"
        )
