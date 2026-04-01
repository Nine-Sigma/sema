from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sema.graph.materializer_utils import (
    apply_resolution_edges as _apply_resolution_edges,
    pick_winner as _pick_winner,
    upsert_column_nodes as _upsert_column_nodes,
    upsert_physical_nodes as _upsert_physical_nodes,
    upsert_semantic_nodes as _upsert_semantic_nodes,
)
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)

logger = logging.getLogger(__name__)


class GraphLoader:
    def __init__(self, driver: Any) -> None:
        self._driver = driver

    def _run(self, cypher: str, **params: Any) -> None:
        with self._driver.session() as session:
            session.run(cypher, **params)

    def _run_read(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    def upsert_datasource(
        self, id: str, ref: str, platform: str, workspace: str,
    ) -> None:
        self._run(
            "MERGE (ds:DataSource {ref: $ref}) "
            "ON CREATE SET ds.id = $id "
            "SET ds.platform = $platform, ds.workspace = $workspace",
            id=id, ref=ref, platform=platform, workspace=workspace,
        )

    def upsert_catalog(self, name: str, ref: str | None = None) -> None:
        id_ = str(uuid.uuid4())
        self._run(
            "MERGE (c:Catalog {name: $name}) "
            "ON CREATE SET c.id = $id, c.ref = $ref",
            name=name, id=id_, ref=ref,
        )

    def upsert_schema(
        self, name: str, catalog: str, ref: str | None = None,
    ) -> None:
        id_ = str(uuid.uuid4())
        self._run(
            "MERGE (c:Catalog {name: $catalog}) "
            "MERGE (s:Schema {name: $name, catalog: $catalog}) "
            "ON CREATE SET s.id = $id, s.ref = $ref "
            "MERGE (s)-[:IN_CATALOG]->(c)",
            name=name, catalog=catalog, id=id_, ref=ref,
        )

    def upsert_table(
        self, name: str, schema_name: str, catalog: str,
        table_type: str = "TABLE", comment: str | None = None,
        ref: str | None = None,
    ) -> None:
        id_ = str(uuid.uuid4())
        self._run(
            "MERGE (s:Schema {name: $schema_name, catalog: $catalog}) "
            "MERGE (t:Table {name: $name, schema_name: $schema_name, "
            "catalog: $catalog}) "
            "ON CREATE SET t.id = $id, t.ref = $ref "
            "SET t.table_type = $table_type, t.comment = $comment "
            "MERGE (t)-[:IN_SCHEMA]->(s)",
            name=name, schema_name=schema_name, catalog=catalog,
            table_type=table_type, comment=comment, id=id_, ref=ref,
        )

    def upsert_column(
        self, name: str, table_name: str, schema_name: str,
        catalog: str, data_type: str, nullable: bool = True,
        comment: str | None = None, ref: str | None = None,
    ) -> None:
        id_ = str(uuid.uuid4())
        self._run(
            "MERGE (t:Table {name: $table_name, "
            "schema_name: $schema_name, catalog: $catalog}) "
            "MERGE (c:Column {name: $name, table_name: $table_name, "
            "schema_name: $schema_name, catalog: $catalog}) "
            "ON CREATE SET c.id = $id, c.ref = $ref "
            "SET c.data_type = $data_type, c.nullable = $nullable, "
            "c.comment = $comment "
            "MERGE (c)-[:IN_TABLE]->(t)",
            name=name, table_name=table_name,
            schema_name=schema_name, catalog=catalog,
            data_type=data_type, nullable=nullable, comment=comment,
            id=id_, ref=ref,
        )

    def batch_upsert_columns(
        self, columns: list[dict[str, Any]], table_name: str,
        schema_name: str, catalog: str,
    ) -> None:
        self._run(
            "UNWIND $columns AS col "
            "MERGE (t:Table {name: $table_name, "
            "schema_name: $schema_name, catalog: $catalog}) "
            "MERGE (c:Column {name: col.name, "
            "table_name: $table_name, "
            "schema_name: $schema_name, catalog: $catalog}) "
            "ON CREATE SET c.id = col.id, c.ref = col.ref "
            "SET c.data_type = col.data_type, "
            "c.nullable = col.nullable, c.comment = col.comment "
            "MERGE (c)-[:IN_TABLE]->(t)",
            columns=columns, table_name=table_name,
            schema_name=schema_name, catalog=catalog,
        )

    def batch_upsert_entities(
        self, entities: list[dict[str, Any]],
    ) -> None:
        from sema.graph import loader_utils as _lu
        _lu.batch_upsert_entities(self, entities)

    def batch_upsert_properties(
        self, properties: list[dict[str, Any]],
    ) -> None:
        from sema.graph import loader_utils as _lu
        _lu.batch_upsert_properties(self, properties)

    def batch_upsert_terms(
        self, terms: list[dict[str, Any]],
    ) -> None:
        from sema.graph import loader_utils as _lu
        _lu.batch_upsert_terms(self, terms)

    def batch_upsert_aliases(
        self, aliases: list[dict[str, Any]], parent_label: str,
    ) -> None:
        from sema.graph import loader_utils as _lu
        _lu.batch_upsert_aliases(self, aliases, parent_label)

    def batch_upsert_value_sets(
        self, value_sets: list[dict[str, Any]],
    ) -> None:
        from sema.graph import loader_utils as _lu
        _lu.batch_upsert_value_sets(self, value_sets)

    def batch_upsert_join_paths(
        self, join_paths: list[dict[str, Any]],
    ) -> None:
        from sema.graph import loader_utils as _lu
        _lu.batch_upsert_join_paths(self, join_paths)

    def upsert_entity(
        self, name: str, description: str | None, source: str,
        confidence: float, table_name: str, schema_name: str,
        catalog: str,
    ) -> None:
        id_ = str(uuid.uuid4())
        self._run(
            "MERGE (e:Entity {name: $name}) "
            "ON CREATE SET e.id = $id "
            "SET e.description = $description, e.source = $source, "
            "e.confidence = $confidence, "
            "e.resolved_at = $resolved_at "
            "WITH e "
            "MERGE (t:Table {name: $table_name, "
            "schema_name: $schema_name, catalog: $catalog}) "
            "MERGE (e)-[:ENTITY_ON_TABLE]->(t)",
            name=name, description=description, source=source,
            confidence=confidence,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            table_name=table_name, schema_name=schema_name,
            catalog=catalog, id=id_,
        )

    def upsert_property(
        self, name: str, semantic_type: str, source: str,
        confidence: float, entity_name: str, column_name: str,
        table_name: str, schema_name: str, catalog: str,
    ) -> None:
        id_ = str(uuid.uuid4())
        self._run(
            "MERGE (p:Property {name: $name, "
            "entity_name: $entity_name}) "
            "ON CREATE SET p.id = $id "
            "SET p.semantic_type = $semantic_type, "
            "p.source = $source, "
            "p.confidence = $confidence, "
            "p.resolved_at = $resolved_at "
            "WITH p "
            "MERGE (e:Entity {name: $entity_name}) "
            "MERGE (e)-[:HAS_PROPERTY]->(p) "
            "WITH p "
            "MERGE (c:Column {name: $column_name, "
            "table_name: $table_name, "
            "schema_name: $schema_name, catalog: $catalog}) "
            "MERGE (p)-[:PROPERTY_ON_COLUMN]->(c)",
            name=name, semantic_type=semantic_type, source=source,
            confidence=confidence,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            entity_name=entity_name, column_name=column_name,
            table_name=table_name, schema_name=schema_name,
            catalog=catalog, id=id_,
        )

    def upsert_term(
        self, code: str, label: str, source: str,
        confidence: float,
    ) -> None:
        id_ = str(uuid.uuid4())
        self._run(
            "MERGE (t:Term {code: $code}) "
            "ON CREATE SET t.id = $id "
            "SET t.label = $label, t.source = $source, "
            "t.confidence = $confidence, "
            "t.resolved_at = $resolved_at",
            code=code, label=label, source=source,
            confidence=confidence,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            id=id_,
        )

    def upsert_value_set(
        self, name: str, column_name: str, table_name: str,
        schema_name: str, catalog: str,
    ) -> None:
        id_ = str(uuid.uuid4())
        self._run(
            "MERGE (vs:ValueSet {name: $name}) "
            "ON CREATE SET vs.id = $id "
            "WITH vs "
            "MERGE (c:Column {name: $column_name, "
            "table_name: $table_name, "
            "schema_name: $schema_name, catalog: $catalog}) "
            "MERGE (c)-[:HAS_VALUE_SET]->(vs)",
            name=name, column_name=column_name,
            table_name=table_name, schema_name=schema_name,
            catalog=catalog, id=id_,
        )

    def add_term_to_value_set(
        self, term_code: str, value_set_name: str,
    ) -> None:
        self._run(
            "MERGE (t:Term {code: $term_code}) "
            "MERGE (vs:ValueSet {name: $value_set_name}) "
            "MERGE (t)-[:MEMBER_OF]->(vs)",
            term_code=term_code, value_set_name=value_set_name,
        )

    def add_term_hierarchy(
        self, parent_code: str, child_code: str,
    ) -> None:
        self._run(
            "MERGE (p:Term {code: $parent_code}) "
            "MERGE (c:Term {code: $child_code}) "
            "MERGE (p)-[:PARENT_OF]->(c)",
            parent_code=parent_code, child_code=child_code,
        )

    def upsert_alias(
        self, text: str, parent_label: str, parent_name: str,
        source: str, confidence: float, is_preferred: bool = False,
        description: str | None = None,
    ) -> None:
        id_ = str(uuid.uuid4())
        self._run(
            f"MERGE (a:Alias {{text: $text}}) "
            f"ON CREATE SET a.id = $id "
            f"SET a.source = $source, a.confidence = $confidence, "
            f"a.resolved_at = $resolved_at, "
            f"a.is_preferred = $is_preferred, "
            f"a.description = $description "
            f"WITH a "
            f"MERGE (p{parent_label} {{name: $parent_name}}) "
            f"MERGE (a)-[:REFERS_TO]->(p)",
            text=text, parent_name=parent_name, source=source,
            confidence=confidence,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            is_preferred=is_preferred, description=description,
            id=id_,
        )

    def upsert_join_path(
        self, name: str, join_predicates: list[dict[str, Any]],
        hop_count: int, source: str, confidence: float,
        sql_snippet: str | None = None,
        cardinality_hint: str | None = None,
    ) -> None:
        id_ = str(uuid.uuid4())
        self._run(
            "MERGE (jp:JoinPath {name: $name}) "
            "ON CREATE SET jp.id = $id "
            "SET jp.join_predicates = $join_predicates, "
            "jp.hop_count = $hop_count, jp.source = $source, "
            "jp.confidence = $confidence, "
            "jp.sql_snippet = $sql_snippet, "
            "jp.cardinality_hint = $cardinality_hint, "
            "jp.resolved_at = $resolved_at",
            name=name,
            join_predicates=json.dumps(join_predicates),
            hop_count=hop_count, source=source,
            confidence=confidence, sql_snippet=sql_snippet,
            cardinality_hint=cardinality_hint,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            id=id_,
        )

    def add_join_path_uses(
        self, join_path_name: str, table_ref: str,
        column_name: str | None = None,
    ) -> None:
        if column_name:
            self._run(
                "MATCH (jp:JoinPath {name: $jp_name}) "
                "MATCH (c:Column {ref: $ref}) "
                "MERGE (jp)-[:USES]->(c)",
                jp_name=join_path_name, ref=table_ref,
            )
        else:
            self._run(
                "MATCH (jp:JoinPath {name: $jp_name}) "
                "MATCH (t:Table {ref: $ref}) "
                "MERGE (jp)-[:USES]->(t)",
                jp_name=join_path_name, ref=table_ref,
            )

    def add_join_path_entity_links(
        self, join_path_name: str, from_table_ref: str,
        to_table_ref: str,
    ) -> None:
        self._run(
            "MATCH (jp:JoinPath {name: $jp_name}) "
            "OPTIONAL MATCH (fe:Entity)-[:ENTITY_ON_TABLE]->"
            "(:Table {ref: $from_ref}) "
            "OPTIONAL MATCH (te:Entity)-[:ENTITY_ON_TABLE]->"
            "(:Table {ref: $to_ref}) "
            "FOREACH (_ IN CASE WHEN fe IS NOT NULL "
            "THEN [1] ELSE [] END | "
            "MERGE (jp)-[:FROM_ENTITY]->(fe)) "
            "FOREACH (_ IN CASE WHEN te IS NOT NULL "
            "THEN [1] ELSE [] END | "
            "MERGE (jp)-[:TO_ENTITY]->(te))",
            jp_name=join_path_name,
            from_ref=from_table_ref,
            to_ref=to_table_ref,
        )

    def store_assertion(self, assertion: Assertion) -> None:
        # NOTE: No longer mutating prior assertions to 'superseded'.
        # Status transitions are now handled via StatusEvent log.
        # Prior assertions remain as immutable history.
        self._run(
            "CREATE (a:Assertion {"
            "  id: $id, subject_ref: $subject_ref,"
            "  subject_id: $subject_id,"
            "  predicate: $predicate,"
            "  payload: $payload, object_ref: $object_ref,"
            "  object_id: $object_id,"
            "  source: $source, confidence: $confidence,"
            "  status: $status, run_id: $run_id,"
            "  observed_at: $observed_at"
            "})",
            id=assertion.id,
            subject_ref=assertion.subject_ref,
            subject_id=assertion.subject_id,
            predicate=assertion.predicate.value,
            payload=json.dumps(assertion.payload),
            object_ref=assertion.object_ref,
            object_id=assertion.object_id,
            source=assertion.source,
            confidence=assertion.confidence,
            status="auto",
            run_id=assertion.run_id,
            observed_at=assertion.observed_at.isoformat(),
        )

    def batch_store_assertions(
        self, assertions: list[Assertion],
    ) -> None:
        for assertion in assertions:
            self.store_assertion(assertion)

    def _build_assertion_dicts(
        self, assertions: list[Assertion],
    ) -> list[dict[str, Any]]:
        return [
            {
                "id": a.id,
                "subject_ref": a.subject_ref,
                "subject_id": a.subject_id,
                "predicate": a.predicate.value,
                "payload": json.dumps(a.payload),
                "object_ref": a.object_ref,
                "object_id": a.object_id,
                "source": a.source,
                "confidence": a.confidence,
                "status": "auto",
                "run_id": a.run_id,
                "observed_at": a.observed_at.isoformat(),
            }
            for a in assertions
        ]

    def commit_table_assertions(
        self, assertions: list[Assertion],
    ) -> None:
        with self._driver.session() as session:
            tx = session.begin_transaction()
            try:
                # NOTE: No longer mutating prior assertions to 'superseded'.
                # Status transitions are now handled via StatusEvent log.
                assertion_dicts = self._build_assertion_dicts(
                    assertions
                )
                tx.run(
                    "UNWIND $assertions AS a "
                    "CREATE (n:Assertion {"
                    "  id: a.id,"
                    "  subject_ref: a.subject_ref,"
                    "  subject_id: a.subject_id,"
                    "  predicate: a.predicate,"
                    "  payload: a.payload,"
                    "  object_ref: a.object_ref,"
                    "  object_id: a.object_id,"
                    "  source: a.source,"
                    "  confidence: a.confidence,"
                    "  status: a.status,"
                    "  run_id: a.run_id,"
                    "  observed_at: a.observed_at"
                    "})",
                    assertions=assertion_dicts,
                )
                tx.commit()
            except Exception:
                tx.rollback()
                raise

    def materialize_provenance_edges(
        self, assertions: list[Assertion],
    ) -> None:
        provenance_predicates = {
            "has_entity_name", "has_property_name", "has_alias",
            "has_semantic_type", "has_decoded_value",
            "vocabulary_match", "parent_of", "has_join_evidence",
        }
        for a in assertions:
            if a.predicate.value not in provenance_predicates:
                continue
            self._run(
                "MATCH (assertion:Assertion {id: $a_id}) "
                "MATCH (n) WHERE n.id = $subject_id "
                "MERGE (assertion)-[:SUBJECT]->(n)",
                a_id=a.id, subject_id=a.subject_id,
            )
            if a.object_id:
                self._run(
                    "MATCH (assertion:Assertion {id: $a_id}) "
                    "MATCH (n) WHERE n.id = $object_id "
                    "MERGE (assertion)-[:OBJECT]->(n)",
                    a_id=a.id, object_id=a.object_id,
                )

    def materialize_table_graph(
        self, assertions: list[Assertion],
    ) -> None:
        """Legacy entry point — delegates to unified materializer."""
        from sema.graph.materializer import materialize_unified
        materialize_unified(self, assertions)

    def query_nodes_by_label(
        self, label: str,
    ) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (n:{label}) RETURN properties(n) AS props",
            )
            return [record["props"] for record in result]

    def set_property_embedding(
        self, name: str, entity_name: str,
        embedding: list[float],
    ) -> None:
        self._run(
            "MATCH (n:Property {name: $name, "
            "entity_name: $entity_name}) "
            "SET n.embedding = $embedding, "
            "n.embedding_updated_at = $updated_at",
            name=name, entity_name=entity_name,
            embedding=embedding,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def set_embedding(
        self, label: str, match_prop: str,
        match_value: str, embedding: list[float],
    ) -> None:
        self._run(
            f"MATCH (n:{label} {{{match_prop}: $match_value}}) "
            f"SET n.embedding = $embedding, "
            f"n.embedding_updated_at = $updated_at",
            match_value=match_value, embedding=embedding,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def has_assertions(self, table_ref: str) -> bool:
        table_ref_slash = table_ref + "/"
        with self._driver.session() as session:
            result = session.run(
                "MATCH (a:Assertion) "
                "WHERE (a.subject_ref = $table_ref "
                "OR a.subject_ref STARTS WITH $table_ref_slash) "
                "AND a.status <> 'superseded' "
                "RETURN count(a) > 0 AS has",
                table_ref=table_ref,
                table_ref_slash=table_ref_slash,
            )
            record = result.single()
            return bool(record["has"]) if record else False

    def load_assertions(
        self, table_ref: str,
    ) -> list[dict[str, Any]]:
        table_ref_slash = table_ref + "/"
        with self._driver.session() as session:
            result = session.run(
                "MATCH (a:Assertion) "
                "WHERE (a.subject_ref = $table_ref "
                "OR a.subject_ref STARTS WITH "
                "$table_ref_slash) "
                "AND a.status <> 'superseded' "
                "RETURN properties(a) AS props",
                table_ref=table_ref,
                table_ref_slash=table_ref_slash,
            )
            return [record["props"] for record in result]

    def create_vector_index(
        self, index_name: str, label: str,
        dimensions: int = 1536,
    ) -> None:
        self._run(
            f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
            f"FOR (n:{label}) ON n.embedding "
            f"OPTIONS {{indexConfig: {{"
            f"  `vector.dimensions`: {dimensions},"
            f"  `vector.similarity_function`: 'cosine'"
            f"}}}}"
        )

    def create_vector_indexes_from_config(
        self, embeddable_labels: list[str],
        dimensions: int = 1536,
    ) -> None:
        for label in embeddable_labels:
            index_name = f"{label.lower()}_embedding_index"
            self.create_vector_index(index_name, label, dimensions)
