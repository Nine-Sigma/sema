from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sema.graph.materializer import (
    _apply_resolution_edges,
    _pick_winner,
    _upsert_column_nodes,
    _upsert_physical_nodes,
    _upsert_semantic_nodes,
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

    def upsert_catalog(self, name: str) -> None:
        self._run(
            "MERGE (c:Catalog {name: $name})",
            name=name,
        )

    def upsert_schema(self, name: str, catalog: str) -> None:
        self._run(
            "MERGE (c:Catalog {name: $catalog}) "
            "MERGE (s:Schema {name: $name, catalog: $catalog}) "
            "MERGE (s)-[:IN_CATALOG]->(c)",
            name=name, catalog=catalog,
        )

    def upsert_table(self, name: str, schema_name: str, catalog: str,
                     table_type: str = "TABLE", comment: str | None = None) -> None:
        self._run(
            "MERGE (s:Schema {name: $schema_name, catalog: $catalog}) "
            "MERGE (t:Table {name: $name, schema_name: $schema_name, catalog: $catalog}) "
            "SET t.table_type = $table_type, t.comment = $comment "
            "MERGE (t)-[:IN_SCHEMA]->(s)",
            name=name, schema_name=schema_name, catalog=catalog,
            table_type=table_type, comment=comment,
        )

    def upsert_column(self, name: str, table_name: str, schema_name: str, catalog: str,
                      data_type: str, nullable: bool = True, comment: str | None = None) -> None:
        self._run(
            "MERGE (t:Table {name: $table_name, schema_name: $schema_name, catalog: $catalog}) "
            "MERGE (c:Column {name: $name, table_name: $table_name, schema_name: $schema_name, catalog: $catalog}) "
            "SET c.data_type = $data_type, c.nullable = $nullable, c.comment = $comment "
            "MERGE (c)-[:IN_TABLE]->(t)",
            name=name, table_name=table_name, schema_name=schema_name, catalog=catalog,
            data_type=data_type, nullable=nullable, comment=comment,
        )

    def batch_upsert_columns(self, columns: list[dict[str, Any]], table_name: str,
                             schema_name: str, catalog: str) -> None:
        self._run(
            "UNWIND $columns AS col "
            "MERGE (t:Table {name: $table_name, schema_name: $schema_name, catalog: $catalog}) "
            "MERGE (c:Column {name: col.name, table_name: $table_name, schema_name: $schema_name, catalog: $catalog}) "
            "SET c.data_type = col.data_type, c.nullable = col.nullable, c.comment = col.comment "
            "MERGE (c)-[:IN_TABLE]->(t)",
            columns=columns, table_name=table_name, schema_name=schema_name, catalog=catalog,
        )

    def batch_upsert_entities(self, entities: list[dict[str, Any]]) -> None:
        from sema.graph import loader_utils as _lu
        _lu.batch_upsert_entities(self, entities)

    def batch_upsert_properties(self, properties: list[dict[str, Any]]) -> None:
        from sema.graph import loader_utils as _lu
        _lu.batch_upsert_properties(self, properties)

    def batch_upsert_terms(self, terms: list[dict[str, Any]]) -> None:
        from sema.graph import loader_utils as _lu
        _lu.batch_upsert_terms(self, terms)

    def batch_upsert_synonyms(self, synonyms: list[dict[str, Any]], parent_label: str) -> None:
        from sema.graph import loader_utils as _lu
        _lu.batch_upsert_synonyms(self, synonyms, parent_label)

    def batch_upsert_value_sets(self, value_sets: list[dict[str, Any]]) -> None:
        from sema.graph import loader_utils as _lu
        _lu.batch_upsert_value_sets(self, value_sets)

    def upsert_entity(self, name: str, description: str | None, source: str,
                      confidence: float, table_name: str, schema_name: str, catalog: str) -> None:
        self._run(
            "MERGE (e:Entity {name: $name}) "
            "SET e.description = $description, e.source = $source, "
            "e.confidence = $confidence, e.resolved_at = $resolved_at "
            "WITH e "
            "MERGE (t:Table {name: $table_name, schema_name: $schema_name, catalog: $catalog}) "
            "MERGE (e)-[:IMPLEMENTED_BY]->(t)",
            name=name, description=description, source=source,
            confidence=confidence, resolved_at=datetime.now(timezone.utc).isoformat(),
            table_name=table_name, schema_name=schema_name, catalog=catalog,
        )

    def upsert_property(self, name: str, semantic_type: str, source: str, confidence: float,
                        entity_name: str, column_name: str, table_name: str,
                        schema_name: str, catalog: str) -> None:
        self._run(
            "MERGE (p:Property {name: $name, entity_name: $entity_name}) "
            "SET p.semantic_type = $semantic_type, p.source = $source, "
            "p.confidence = $confidence, p.resolved_at = $resolved_at "
            "WITH p "
            "MERGE (e:Entity {name: $entity_name}) "
            "MERGE (e)-[:HAS_PROPERTY]->(p) "
            "WITH p "
            "MERGE (c:Column {name: $column_name, table_name: $table_name, "
            "schema_name: $schema_name, catalog: $catalog}) "
            "MERGE (p)-[:IMPLEMENTED_BY]->(c)",
            name=name, semantic_type=semantic_type, source=source,
            confidence=confidence, resolved_at=datetime.now(timezone.utc).isoformat(),
            entity_name=entity_name, column_name=column_name,
            table_name=table_name, schema_name=schema_name, catalog=catalog,
        )

    def upsert_term(self, code: str, label: str, source: str, confidence: float) -> None:
        self._run(
            "MERGE (t:Term {code: $code}) "
            "SET t.label = $label, t.source = $source, "
            "t.confidence = $confidence, t.resolved_at = $resolved_at",
            code=code, label=label, source=source,
            confidence=confidence, resolved_at=datetime.now(timezone.utc).isoformat(),
        )

    def upsert_value_set(self, name: str, column_name: str, table_name: str,
                         schema_name: str, catalog: str) -> None:
        self._run(
            "MERGE (vs:ValueSet {name: $name}) "
            "WITH vs "
            "MERGE (c:Column {name: $column_name, table_name: $table_name, "
            "schema_name: $schema_name, catalog: $catalog}) "
            "MERGE (vs)-[:STORED_IN]->(c)",
            name=name, column_name=column_name, table_name=table_name,
            schema_name=schema_name, catalog=catalog,
        )

    def add_term_to_value_set(self, term_code: str, value_set_name: str) -> None:
        self._run(
            "MERGE (t:Term {code: $term_code}) "
            "MERGE (vs:ValueSet {name: $value_set_name}) "
            "MERGE (t)-[:MEMBER_OF]->(vs)",
            term_code=term_code, value_set_name=value_set_name,
        )

    def add_term_hierarchy(self, parent_code: str, child_code: str) -> None:
        self._run(
            "MERGE (p:Term {code: $parent_code}) "
            "MERGE (c:Term {code: $child_code}) "
            "MERGE (p)-[:PARENT_OF]->(c)",
            parent_code=parent_code, child_code=child_code,
        )

    def upsert_synonym(self, text: str, parent_label: str, parent_name: str,
                       source: str, confidence: float) -> None:
        self._run(
            f"MERGE (s:Synonym {{text: $text}}) "
            f"SET s.source = $source, s.confidence = $confidence, "
            f"s.resolved_at = $resolved_at "
            f"WITH s "
            f"MERGE (p{parent_label} {{name: $parent_name}}) "
            f"MERGE (s)-[:SYNONYM_OF]->(p)",
            text=text, parent_name=parent_name, source=source,
            confidence=confidence, resolved_at=datetime.now(timezone.utc).isoformat(),
        )

    def upsert_candidate_join(self, from_table: str, from_schema: str, from_catalog: str,
                              to_table: str, to_schema: str, to_catalog: str,
                              on_column: str, cardinality: str,
                              source: str, confidence: float,
                              semantic_label: str | None = None) -> None:
        self._run(
            "MERGE (f:Table {name: $from_table, schema_name: $from_schema, catalog: $from_catalog}) "
            "MERGE (t:Table {name: $to_table, schema_name: $to_schema, catalog: $to_catalog}) "
            "MERGE (f)-[j:CANDIDATE_JOIN {on_column: $on_column}]->(t) "
            "SET j.cardinality = $cardinality, j.source = $source, "
            "j.confidence = $confidence, j.semantic_label = $semantic_label",
            from_table=from_table, from_schema=from_schema, from_catalog=from_catalog,
            to_table=to_table, to_schema=to_schema, to_catalog=to_catalog,
            on_column=on_column, cardinality=cardinality,
            source=source, confidence=confidence, semantic_label=semantic_label,
        )

    def store_assertion(self, assertion: Assertion) -> None:
        # Supersede old assertions with same dedupe key
        # Only supersede 'auto' status assertions — human overrides are preserved
        self._run(
            "MATCH (a:Assertion) "
            "WHERE a.subject_ref = $subject_ref "
            "AND a.predicate = $predicate "
            "AND a.source = $source "
            "AND a.status = 'auto' "
            "AND a.run_id <> $run_id "
            "SET a.status = 'superseded'",
            subject_ref=assertion.subject_ref,
            predicate=assertion.predicate.value,
            source=assertion.source,
            run_id=assertion.run_id,
        )

        self._run(
            "CREATE (a:Assertion {"
            "  id: $id, subject_ref: $subject_ref, predicate: $predicate,"
            "  payload: $payload, object_ref: $object_ref,"
            "  source: $source, confidence: $confidence,"
            "  status: $status, run_id: $run_id, observed_at: $observed_at"
            "})",
            id=assertion.id,
            subject_ref=assertion.subject_ref,
            predicate=assertion.predicate.value,
            payload=json.dumps(assertion.payload),
            object_ref=assertion.object_ref,
            source=assertion.source,
            confidence=assertion.confidence,
            status=assertion.status.value,
            run_id=assertion.run_id,
            observed_at=assertion.observed_at.isoformat(),
        )

    def batch_store_assertions(self, assertions: list[Assertion]) -> None:
        for assertion in assertions:
            self.store_assertion(assertion)

    def _build_supersession_groups(
        self, assertions: list[Assertion],
    ) -> dict[tuple[str, str, str], str]:
        groups: dict[tuple[str, str, str], str] = {}
        for a in assertions:
            groups[(a.subject_ref, a.predicate.value, a.source)] = a.run_id
        return groups

    def _build_assertion_dicts(
        self, assertions: list[Assertion]
    ) -> list[dict[str, Any]]:
        return [
            {
                "id": a.id,
                "subject_ref": a.subject_ref,
                "predicate": a.predicate.value,
                "payload": json.dumps(a.payload),
                "object_ref": a.object_ref,
                "source": a.source,
                "confidence": a.confidence,
                "status": a.status.value,
                "run_id": a.run_id,
                "observed_at": a.observed_at.isoformat(),
            }
            for a in assertions
        ]

    def commit_table_assertions(
        self, assertions: list[Assertion]
    ) -> None:
        """Phase 1: Commit all assertions for a table in a single transaction.

        Supersession + creation are atomic — either all succeed or all
        roll back.
        """
        with self._driver.session() as session:
            tx = session.begin_transaction()
            try:
                groups = self._build_supersession_groups(assertions)

                for (subj, pred, src), run_id in groups.items():
                    tx.run(
                        "MATCH (a:Assertion) "
                        "WHERE a.subject_ref = $subject_ref "
                        "AND a.predicate = $predicate "
                        "AND a.source = $source "
                        "AND a.status = 'auto' "
                        "AND a.run_id <> $run_id "
                        "SET a.status = 'superseded'",
                        subject_ref=subj,
                        predicate=pred,
                        source=src,
                        run_id=run_id,
                    )

                # UNWIND batch create
                assertion_dicts = self._build_assertion_dicts(assertions)
                tx.run(
                    "UNWIND $assertions AS a "
                    "CREATE (n:Assertion {"
                    "  id: a.id,"
                    "  subject_ref: a.subject_ref,"
                    "  predicate: a.predicate,"
                    "  payload: a.payload,"
                    "  object_ref: a.object_ref,"
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

    def materialize_table_graph(self, assertions: list[Assertion]) -> None:
        """Phase 2: Derive and upsert graph nodes from committed assertions.

        Deterministic and idempotent (uses MERGE). Can be re-run safely.
        """
        by_subject: dict[str, list[Assertion]] = defaultdict(list)
        for a in assertions:
            by_subject[a.subject_ref].append(a)
        groups: dict[tuple[str, str], list[Assertion]] = defaultdict(list)
        for a in assertions:
            groups[(a.subject_ref, a.predicate.value)].append(a)

        _upsert_physical_nodes(self, by_subject)
        _upsert_column_nodes(self, by_subject)
        _upsert_semantic_nodes(self, by_subject, groups)
        _apply_resolution_edges(self, groups)

    def query_nodes_by_label(self, label: str) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (n:{label}) RETURN properties(n) AS props",
            )
            return [record["props"] for record in result]

    def set_property_embedding(
        self, name: str, entity_name: str, embedding: list[float],
    ) -> None:
        self._run(
            "MATCH (n:Property {name: $name, entity_name: $entity_name}) "
            "SET n.embedding = $embedding",
            name=name, entity_name=entity_name, embedding=embedding,
        )

    def set_embedding(self, label: str, match_prop: str, match_value: str,
                      embedding: list[float]) -> None:
        self._run(
            f"MATCH (n:{label} {{{match_prop}: $match_value}}) "
            f"SET n.embedding = $embedding",
            match_value=match_value, embedding=embedding,
        )

    def has_assertions(self, table_ref: str) -> bool:
        """Check whether non-superseded assertions exist for *table_ref*."""
        table_ref_dot = table_ref + "."
        with self._driver.session() as session:
            result = session.run(
                "MATCH (a:Assertion) "
                "WHERE (a.subject_ref = $table_ref "
                "OR a.subject_ref STARTS WITH $table_ref_dot) "
                "AND a.status <> 'superseded' "
                "RETURN count(a) > 0 AS has",
                table_ref=table_ref,
                table_ref_dot=table_ref_dot,
            )
            record = result.single()
            return bool(record["has"]) if record else False

    def load_assertions(self, table_ref: str) -> list[dict[str, Any]]:
        """Load all non-superseded assertion dicts for *table_ref*."""
        table_ref_dot = table_ref + "."
        with self._driver.session() as session:
            result = session.run(
                "MATCH (a:Assertion) "
                "WHERE (a.subject_ref = $table_ref "
                "OR a.subject_ref STARTS WITH $table_ref_dot) "
                "AND a.status <> 'superseded' "
                "RETURN properties(a) AS props",
                table_ref=table_ref,
                table_ref_dot=table_ref_dot,
            )
            return [record["props"] for record in result]

    def create_vector_index(self, index_name: str, label: str, dimensions: int = 1536) -> None:
        self._run(
            f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
            f"FOR (n:{label}) ON n.embedding "
            f"OPTIONS {{indexConfig: {{"
            f"  `vector.dimensions`: {dimensions},"
            f"  `vector.similarity_function`: 'cosine'"
            f"}}}}"
        )
