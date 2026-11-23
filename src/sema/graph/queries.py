from __future__ import annotations

class CypherQueries:
    """Parameterized Cypher query templates for the semantic graph."""

    @staticmethod
    def _vector_read_factor() -> int:
        # An approximate (HNSW, quantized) index can misplace exact neighbours near its cut.
        return 3

    @staticmethod
    def vector_search(index_name: str, top_k: int = 10) -> str:
        width = top_k * CypherQueries._vector_read_factor()
        return (
            f"CALL db.index.vector.queryNodes('{index_name}', {width}, $embedding) "
            f"YIELD node "
            f"WITH node, vector.similarity.cosine(node.embedding, $embedding) AS score "
            f"RETURN node, score ORDER BY score DESC, node.ref, node.schema_name, "
            f"node.entity_name, node.name, node.vocabulary_name, node.code, "
            f"node.target_key, node.text LIMIT {top_k}"
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
    def expand_hierarchy_anchor(max_depth: int = 3) -> str:
        # A column member sits under the anchor directly or through a
        # MAPS_TO_CONCEPT bridge from a source code to a standard descendant.
        return (
            f"MATCH (anchor:Term)-[:PARENT_OF*1..{max_depth}]->(d:Term) "
            f"WHERE anchor.code = $code "
            f"AND (anchor.vocabulary_name = $vocabulary_name "
            f"OR $vocabulary_name IS NULL) "
            f"WITH DISTINCT anchor, d "
            f"OPTIONAL MATCH (src:Term)-[:MAPS_TO_CONCEPT]->(d) "
            f"WITH anchor, d, collect(src) AS sources "
            f"UNWIND [d] + sources AS m "
            f"MATCH (m)-[:MEMBER_OF]->(:ValueSet)<-[:HAS_VALUE_SET]-(c:Column) "
            f"OPTIONAL MATCH (c)-[:IN_TABLE]->(tbl:Table) "
            f"WITH DISTINCT anchor, c, tbl, m ORDER BY m.code "
            f"WITH anchor, c, tbl, collect(m.code) AS members "
            f"RETURN anchor.code AS anchor_code, anchor.label AS anchor_label, "
            f"anchor.vocabulary_name AS anchor_vocabulary, "
            f"c.name AS column_name, "
            f"coalesce(tbl.name, c.table_name) AS table_name, "
            f"coalesce(tbl.schema_name, c.schema_name) AS schema_name, "
            f"size(members) AS member_count, "
            f"members[..$member_cap] AS member_codes "
            f"ORDER BY table_name, column_name"
        )

    @staticmethod
    def expand_value_set() -> str:
        # Value-set names are not globally unique (same table.column name
        # across studies), so callers that can should prefer
        # find_value_set_members_by_column (column-scoped). vocabulary_name
        # is returned here so same-code terms remain distinguishable.
        return (
            "MATCH (t:Term)-[:MEMBER_OF]->(vs:ValueSet {name: $value_set_name}) "
            "RETURN t.code AS code, t.label AS label, "
            "t.vocabulary_name AS vocabulary_name"
        )

    @staticmethod
    def _physical_mapping_tail() -> str:
        # Every edge is returned, one with `owner = false` too:
        # the retrieval reader drops it.
        return (
            "OPTIONAL MATCH (e)-[:HAS_PROPERTY]->(p:Property)"
            "-[:PROPERTY_ON_COLUMN]->(c:Column)-[:IN_TABLE]->(t) "
            "WITH t, r.owner AS owner, r.key_columns AS key_columns, "
            "r.verification AS verification, r.identity AS identity, "
            "r.null_key_rows AS null_key_rows, "
            "collect({property: p.name, column: c.name, "
            "data_type: c.data_type, semantic_type: p.semantic_type, "
            "labels: CASE WHEN c IS NULL THEN [] ELSE "
            "[(c)-[l:LABELED_BY]->(lc:Column)-[:IN_TABLE]->(lt:Table) | "
            "{catalog: lt.catalog, schema: lt.schema_name, table: lt.name, column: lc.name, "
            "verification: l.verification, "
            "via: [(jp:JoinPath)-[:USES]->(lt) WHERE jp.name IN coalesce(l.via, []) "
            "| jp.join_predicates]}] END}) AS columns "
            "RETURN t.name AS table_name, t.schema_name AS schema_name, "
            "t.catalog AS catalog, columns, "
            "t.key_columns AS table_key_columns, "
            "t.key_verification AS table_key_verification, "
            "owner, key_columns, verification, identity, null_key_rows, "
            "[(o:Entity)-[s:ENTITY_ON_TABLE]->(t) "
            "WHERE s.key_columns IS NOT NULL AND coalesce(s.owner, false) = false | "
            "{entity: o.name, key_columns: s.key_columns, verification: s.verification, "
            "identity: s.identity, "
            "owners: [(o)-[w:ENTITY_ON_TABLE]->(ot:Table) WHERE w.owner = true | "
            "{catalog: ot.catalog, schema: ot.schema_name, table: ot.name}]}] "
            "AS shared_entities"
        )

    @staticmethod
    def resolve_physical_mapping() -> str:
        return (
            "MATCH (e:Entity {name: $entity_name})-[r:ENTITY_ON_TABLE]->(t:Table) "
            + CypherQueries._physical_mapping_tail()
        )

    @staticmethod
    def resolve_physical_mapping_scoped() -> str:
        # M6: with source + target layers coexisting, a shared entity name
        # returns mixed bindings. Scope by model_role / schema_name so a
        # scoped query resolves to its own layer's physical binding. Both
        # filters are OR-null so a caller may constrain either or neither.
        return (
            "MATCH (e:Entity {name: $entity_name})-[r:ENTITY_ON_TABLE]->(t:Table) "
            "WHERE (e.model_role = $model_role OR $model_role IS NULL) "
            "AND (e.schema_name = $schema_name OR $schema_name IS NULL) "
            + CypherQueries._physical_mapping_tail()
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
            "jp.confidence AS confidence, "
            "jp.source_schema AS source_schema"
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
            "AND (t.vocabulary_name = $vocabulary_name "
            "OR $vocabulary_name IS NULL) "
            "OPTIONAL MATCH (c)-[:IN_TABLE]->(tbl:Table) "
            "RETURN vs.name AS value_set_name, "
            "c.name AS column_name, "
            "tbl.name AS table_name, "
            "tbl.schema_name AS schema_name, "
            "tbl.catalog AS catalog"
        )

    @staticmethod
    def find_vocabularies_for_term() -> str:
        # Codes are unique only within a vocabulary, so this returns ALL
        # vocabularies for a code: the caller decides how to handle
        # ambiguity instead of the database picking one arbitrarily.
        return (
            "MATCH (t:Term {code: $code})"
            "-[:IN_VOCABULARY]->(v:Vocabulary) "
            "RETURN DISTINCT v.name AS vocabulary_name "
            "ORDER BY vocabulary_name"
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
        # Scope the column by schema so same-named columns in different
        # studies don't merge, and surface vocabulary_name so callers can
        # disambiguate same-code terms across vocabularies. schema_name is
        # optional (OR-null) to stay backward-compatible with callers that
        # cannot resolve it.
        return (
            "MATCH (c:Column {name: $column_name, "
            "table_name: $table_name}) "
            "WHERE (c.schema_name = $schema_name OR $schema_name IS NULL) "
            "MATCH (c)-[:HAS_VALUE_SET]->(vs:ValueSet) "
            "MATCH (t:Term)-[:MEMBER_OF]->(vs) "
            "RETURN t.code AS code, t.label AS label, "
            "t.vocabulary_name AS vocabulary_name"
        )

    @staticmethod
    def resolve_concept_for_source_term() -> str:
        # Cross-layer bridge (Finding 1): a query seeded on a SOURCE term
        # surfaces the target concept it maps to plus the governed target
        # Column/Table to filter. Matches on MAPS_TO_CONCEPT, not any domain
        # literal. schema_name distinguishes same-named columns across schemas.
        return (
            "MATCH (src:Term {code: $code})-[:MAPS_TO_CONCEPT]->(concept:Term) "
            "WHERE ($source_vocabulary IS NULL "
            "OR src.vocabulary_name = $source_vocabulary) "
            "MATCH (concept)-[:MEMBER_OF]->(vs:ValueSet)"
            "<-[:HAS_VALUE_SET]-(c:Column) "
            "OPTIONAL MATCH (c)-[:IN_TABLE]->(t:Table) "
            "RETURN concept.code AS concept_code, "
            "concept.label AS concept_label, "
            "concept.vocabulary_name AS concept_vocabulary, "
            "c.name AS column_name, c.table_name AS table_name, "
            "c.schema_name AS schema_name, c.catalog AS catalog"
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
