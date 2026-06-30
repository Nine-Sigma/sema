from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Mapping

from sema.graph.materializer_utils import (
    apply_resolution_edges as _apply_resolution_edges,
    pick_winner as _pick_winner,
    upsert_column_nodes as _upsert_column_nodes,
    upsert_physical_nodes as _upsert_physical_nodes,
    upsert_semantic_nodes as _upsert_semantic_nodes,
)
from sema.graph.retry_utils import run_with_retry, statement_summary
from sema.graph.term_identity_utils import term_namespace
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)

logger = logging.getLogger(__name__)

# Idempotent assertion write. MERGE on the immutable assertion id makes the
# write a no-op on replay, so a retried commit (or a duplicate call) can never
# create a second node. ON CREATE SET avoids overwriting any later edits to an
# existing assertion. Shared by store_assertion and commit_table_assertions so
# both production and single-write paths are idempotent.
_ASSERTION_UPSERT_CYPHER = (
    "UNWIND $assertions AS a "
    "MERGE (n:Assertion {id: a.id}) "
    "ON CREATE SET "
    "  n.subject_ref = a.subject_ref, "
    "  n.subject_id = a.subject_id, "
    "  n.predicate = a.predicate, "
    "  n.payload = a.payload, "
    "  n.object_ref = a.object_ref, "
    "  n.object_id = a.object_id, "
    "  n.source = a.source, "
    "  n.confidence = a.confidence, "
    "  n.status = a.status, "
    "  n.run_id = a.run_id, "
    "  n.observed_at = a.observed_at, "
    "  n.source_schema = a.source_schema"
)


class GraphLoader:
    def __init__(self, driver: Any) -> None:
        self._driver = driver

    def _run(self, cypher: str, **params: Any) -> None:
        def op() -> None:
            with self._driver.session() as session:
                session.run(cypher, **params)

        run_with_retry(op, statement_summary(cypher))

    def _run_read(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        def op() -> list[dict[str, Any]]:
            with self._driver.session() as session:
                result = session.run(cypher, **params)
                return [dict(record) for record in result]

        return run_with_retry(op, statement_summary(cypher))

    def ensure_core_constraints(self) -> None:
        """Create core graph constraints, deduping any violations first.

        Idempotent; run once per build before table workers commit. The
        Assertion.id uniqueness constraint both prevents duplicate
        assertions and backs the MERGE in _ASSERTION_UPSERT_CYPHER with an
        index. Pre-existing duplicate-id assertions (e.g. from pre-MERGE
        builds) are collapsed to one node first, because CREATE CONSTRAINT
        fails on already-violating data.
        """
        self._run(
            "MATCH (a:Assertion) "
            "WHERE a.id IS NOT NULL "
            "WITH a.id AS id, collect(a) AS nodes "
            "WHERE size(nodes) > 1 "
            "UNWIND nodes[1..] AS dup "
            "DETACH DELETE dup"
        )
        self._run(
            "CREATE CONSTRAINT assertion_id_unique IF NOT EXISTS "
            "FOR (a:Assertion) REQUIRE a.id IS UNIQUE"
        )

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
        source_schema: str | None = None,
    ) -> None:
        from sema.graph import loader_utils as _lu
        _lu.batch_upsert_terms(self, terms, source_schema=source_schema)

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
        catalog: str, source_schema: str | None = None,
    ) -> None:
        id_ = str(uuid.uuid4())
        self._run(
            "MERGE (e:Entity {name: $name}) "
            "ON CREATE SET e.id = $id "
            "SET e.description = $description, e.source = $source, "
            "e.confidence = $confidence, "
            "e.resolved_at = $resolved_at, "
            "e.model_role = coalesce(e.model_role, 'SOURCE'), "
            "e.source_id = coalesce(e.source_id, $source_schema, $source) "
            "WITH e "
            "MERGE (t:Table {name: $table_name, "
            "schema_name: $schema_name, catalog: $catalog}) "
            "MERGE (e)-[link:ENTITY_ON_TABLE "
            "{source_schema: $source_schema}]->(t)",
            name=name, description=description, source=source,
            confidence=confidence,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            table_name=table_name, schema_name=schema_name,
            catalog=catalog, id=id_, source_schema=source_schema,
        )

    def upsert_property(
        self, name: str, semantic_type: str, source: str,
        confidence: float, entity_name: str, column_name: str,
        table_name: str, schema_name: str, catalog: str,
        source_schema: str | None = None,
    ) -> None:
        id_ = str(uuid.uuid4())
        self._run(
            "MERGE (p:Property {entity_name: $entity_name, "
            "name: $name}) "
            "ON CREATE SET p.id = $id "
            "SET p.semantic_type = $semantic_type, "
            "p.source = $source, "
            "p.confidence = $confidence, "
            "p.resolved_at = $resolved_at, "
            "p.model_role = coalesce(p.model_role, 'SOURCE'), "
            "p.source_id = coalesce(p.source_id, $source_schema, $source) "
            "WITH p "
            "MERGE (e:Entity {name: $entity_name}) "
            "SET e.model_role = coalesce(e.model_role, 'SOURCE'), "
            "e.source_id = coalesce(e.source_id, $source_schema, $source) "
            "MERGE (e)-[hp:HAS_PROPERTY "
            "{source_schema: $source_schema}]->(p) "
            "WITH p "
            "MERGE (c:Column {name: $column_name, "
            "table_name: $table_name, "
            "schema_name: $schema_name, catalog: $catalog}) "
            "MERGE (p)-[poc:PROPERTY_ON_COLUMN "
            "{source_schema: $source_schema}]->(c)",
            name=name, semantic_type=semantic_type, source=source,
            confidence=confidence,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            entity_name=entity_name, column_name=column_name,
            table_name=table_name, schema_name=schema_name,
            catalog=catalog, id=id_, source_schema=source_schema,
        )

    def upsert_term(
        self, code: str, label: str, source: str,
        confidence: float,
        source_schema: str | None = None,
        vocabulary_name: str | None = None,
    ) -> None:
        id_ = str(uuid.uuid4())
        self._run(
            "MERGE (t:Term {vocabulary_name: $vocabulary_name, code: $code}) "
            "ON CREATE SET t.id = $id "
            "SET t.label = $label, t.source = $source, "
            "t.confidence = $confidence, "
            "t.resolved_at = $resolved_at, "
            "t.model_role = coalesce(t.model_role, 'SOURCE'), "
            "t.source_id = coalesce(t.source_id, $source_schema, $source)",
            code=code, label=label, source=source,
            confidence=confidence,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            id=id_, source_schema=source_schema,
            vocabulary_name=term_namespace(vocabulary_name),
        )

    def upsert_value_set(
        self, name: str, column_name: str, table_name: str,
        schema_name: str, catalog: str,
        source_schema: str | None = None,
        column_ref: str | None = None,
    ) -> None:
        id_ = str(uuid.uuid4())
        ref = column_ref or (
            f"{catalog}.{schema_name}.{table_name}.{column_name}"
        )
        self._run(
            "MERGE (vs:ValueSet {column_ref: $column_ref}) "
            "ON CREATE SET vs.id = $id "
            "SET vs.name = $name "
            "WITH vs "
            "MERGE (c:Column {name: $column_name, "
            "table_name: $table_name, "
            "schema_name: $schema_name, catalog: $catalog}) "
            "MERGE (c)-[hvs:HAS_VALUE_SET "
            "{source_schema: $source_schema}]->(vs)",
            name=name, column_name=column_name,
            table_name=table_name, schema_name=schema_name,
            catalog=catalog, id=id_, column_ref=ref,
            source_schema=source_schema,
        )

    def add_term_to_value_set(
        self, term_code: str, value_set_name: str,
        source_schema: str | None = None,
        vocabulary_name: str | None = None,
        value_set_ref: str | None = None,
    ) -> None:
        """Link a Term to its ValueSet via MEMBER_OF.

        When ``value_set_ref`` (a column_ref) is given, the ValueSet is
        matched on ``column_ref`` — the canonical identity used by
        ``upsert_value_set`` / ``batch_upsert_value_sets``. Matching on
        ``name`` (the legacy fallback) would attach MEMBER_OF to a separate
        name-keyed node distinct from the column_ref node that HAS_VALUE_SET
        points at. Production always passes ``value_set_ref``.
        """
        if value_set_ref is not None:
            vs_clause = (
                "MERGE (vs:ValueSet {column_ref: $value_set_ref}) "
                "ON CREATE SET vs.name = $value_set_name "
            )
        else:
            vs_clause = "MERGE (vs:ValueSet {name: $value_set_name}) "
        self._run(
            "MERGE (t:Term {vocabulary_name: $vocabulary_name, "
            "code: $term_code}) "
            + vs_clause +
            "MERGE (t)-[m:MEMBER_OF "
            "{source_schema: $source_schema}]->(vs)",
            term_code=term_code, value_set_name=value_set_name,
            value_set_ref=value_set_ref,
            source_schema=source_schema,
            vocabulary_name=term_namespace(vocabulary_name),
        )

    def add_term_hierarchy(
        self, parent_code: str, child_code: str,
        source_schema: str | None = None,
        vocabulary_name: str | None = None,
    ) -> None:
        self._run(
            "MERGE (p:Term {vocabulary_name: $vocabulary_name, "
            "code: $parent_code}) "
            "MERGE (c:Term {vocabulary_name: $vocabulary_name, "
            "code: $child_code}) "
            "MERGE (p)-[po:PARENT_OF "
            "{source_schema: $source_schema}]->(c)",
            parent_code=parent_code, child_code=child_code,
            source_schema=source_schema,
            vocabulary_name=term_namespace(vocabulary_name),
        )

    def upsert_alias(
        self, text: str, parent_label: str, parent_name: str,
        source: str, confidence: float, is_preferred: bool = False,
        description: str | None = None,
        source_schema: str | None = None,
        parent_entity_name: str | None = None,
    ) -> None:
        id_ = str(uuid.uuid4())
        if parent_label == ":Property":
            parent_match = (
                "MERGE (p:Property {entity_name: $parent_entity_name, "
                "name: $parent_name})"
            )
        else:
            parent_match = (
                f"MERGE (p{parent_label} {{name: $parent_name}})"
            )
        self._run(
            "MERGE (a:Alias {text: $text}) "
            "ON CREATE SET a.id = $id "
            "SET a.source = $source, a.confidence = $confidence, "
            "a.resolved_at = $resolved_at, "
            "a.is_preferred = $is_preferred, "
            "a.description = $description "
            "WITH a "
            f"{parent_match} "
            "MERGE (a)-[ref:REFERS_TO "
            "{source_schema: $source_schema}]->(p)",
            text=text, parent_name=parent_name, source=source,
            confidence=confidence,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            is_preferred=is_preferred, description=description,
            id=id_, source_schema=source_schema,
            parent_entity_name=parent_entity_name,
        )

    def upsert_join_path(
        self, name: str, join_predicates: list[dict[str, Any]],
        hop_count: int, source: str, confidence: float,
        sql_snippet: str | None = None,
        cardinality_hint: str | None = None,
        source_schema: str | None = None,
    ) -> None:
        id_ = str(uuid.uuid4())
        self._run(
            "MERGE (jp:JoinPath {name: $name, "
            "source_schema: $source_schema}) "
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
            id=id_, source_schema=source_schema,
        )

    def add_join_path_uses(
        self, join_path_name: str, table_ref: str,
        column_name: str | None = None,
        source_schema: str | None = None,
    ) -> None:
        if source_schema is None:
            raise ValueError(
                "add_join_path_uses requires source_schema to scope "
                "the JoinPath match by {name, source_schema}"
            )
        if column_name:
            self._run(
                "MATCH (jp:JoinPath {name: $jp_name, "
                "source_schema: $source_schema}) "
                "MATCH (c:Column {ref: $ref}) "
                "MERGE (jp)-[u:USES "
                "{source_schema: $source_schema}]->(c)",
                jp_name=join_path_name, ref=table_ref,
                source_schema=source_schema,
            )
        else:
            self._run(
                "MATCH (jp:JoinPath {name: $jp_name, "
                "source_schema: $source_schema}) "
                "MATCH (t:Table {ref: $ref}) "
                "MERGE (jp)-[u:USES "
                "{source_schema: $source_schema}]->(t)",
                jp_name=join_path_name, ref=table_ref,
                source_schema=source_schema,
            )

    def add_join_path_entity_links(
        self, join_path_name: str, from_table_ref: str,
        to_table_ref: str,
        source_schema: str | None = None,
    ) -> None:
        if source_schema is None:
            raise ValueError(
                "add_join_path_entity_links requires source_schema "
                "to scope the JoinPath match by {name, source_schema}"
            )
        self._run(
            "MATCH (jp:JoinPath {name: $jp_name, "
            "source_schema: $source_schema}) "
            "OPTIONAL MATCH (fe:Entity)-[:ENTITY_ON_TABLE]->"
            "(:Table {ref: $from_ref}) "
            "OPTIONAL MATCH (te:Entity)-[:ENTITY_ON_TABLE]->"
            "(:Table {ref: $to_ref}) "
            "FOREACH (_ IN CASE WHEN fe IS NOT NULL "
            "THEN [1] ELSE [] END | "
            "MERGE (jp)-[fr:FROM_ENTITY "
            "{source_schema: $source_schema}]->(fe)) "
            "FOREACH (_ IN CASE WHEN te IS NOT NULL "
            "THEN [1] ELSE [] END | "
            "MERGE (jp)-[to_:TO_ENTITY "
            "{source_schema: $source_schema}]->(te))",
            jp_name=join_path_name,
            from_ref=from_table_ref,
            to_ref=to_table_ref,
            source_schema=source_schema,
        )

    def store_assertion(
        self, assertion: Assertion, source_schema: str | None = None,
    ) -> None:
        dicts = self._build_assertion_dicts(
            [assertion], source_schema=source_schema,
        )
        self._run(_ASSERTION_UPSERT_CYPHER, assertions=dicts)

    def batch_store_assertions(
        self, assertions: list[Assertion],
        source_schema: str | None = None,
    ) -> None:
        for assertion in assertions:
            self.store_assertion(assertion, source_schema=source_schema)

    def _build_assertion_dicts(
        self, assertions: list[Assertion],
        source_schema: str | None = None,
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
                "source_schema": source_schema or a.source_schema,
            }
            for a in assertions
        ]

    def commit_table_assertions(
        self, assertions: list[Assertion],
        source_schema: str | None = None,
    ) -> None:
        with self._driver.session() as session:
            tx = session.begin_transaction()
            try:
                assertion_dicts = self._build_assertion_dicts(
                    assertions, source_schema=source_schema,
                )
                tx.run(_ASSERTION_UPSERT_CYPHER, assertions=assertion_dicts)
                tx.commit()
            except Exception:
                tx.rollback()
                raise

    def materialize_provenance_edges(
        self, assertions: list[Assertion],
    ) -> None:
        from sema.graph import provenance_utils as _pu
        _pu.materialize_provenance_edges(self, assertions)

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

    def set_node_embedding(
        self, label: str, match: Mapping[str, str],
        embedding: list[float], description_hash: str = "",
    ) -> None:
        """Set an embedding on a node matched by a composite key.

        ``match`` maps node property -> value; ALL pairs must match, so a
        composite-identity node (e.g. ``:Term {vocabulary_name, code}``)
        is addressed unambiguously. Label and property names are validated
        against the embedding allowlist before interpolation.
        """
        from sema.graph.embedding_match import (
            EMBEDDING_MATCH_KEYS,
            validate_match_props,
        )

        if not match:
            raise ValueError(
                "set_node_embedding requires at least one match property"
            )
        validate_match_props(label, match.keys())
        expected = set(EMBEDDING_MATCH_KEYS[label])
        if set(match) != expected:
            raise ValueError(
                f"{label} embedding must match on the full composite key "
                f"{sorted(expected)}, got {sorted(match)}. A partial key "
                "(e.g. Term code without vocabulary_name) would clobber "
                "the embeddings of other nodes sharing that value."
            )
        pattern = ", ".join(f"{prop}: $m_{prop}" for prop in match)
        params = {f"m_{prop}": value for prop, value in match.items()}
        self._run(
            f"MATCH (n:{label} {{{pattern}}}) "
            "SET n.embedding = $embedding, "
            "n.embedding_updated_at = $updated_at, "
            "n.description_hash = $description_hash",
            embedding=embedding, description_hash=description_hash,
            updated_at=datetime.now(timezone.utc).isoformat(),
            **params,
        )

    def set_property_embedding(
        self, name: str, entity_name: str,
        embedding: list[float], description_hash: str = "",
    ) -> None:
        self.set_node_embedding(
            "Property", {"name": name, "entity_name": entity_name},
            embedding, description_hash,
        )

    def set_embedding(
        self, label: str, match_prop: str,
        match_value: str, embedding: list[float],
        description_hash: str = "",
    ) -> None:
        self.set_node_embedding(
            label, {match_prop: match_value}, embedding, description_hash,
        )

    def delete_study_scoped(
        self, schema_name: str, preserve_assertions: bool = False,
    ) -> None:
        """Remove every graph element stamped with this study's schema.

        Edge sweep matches by `source_schema`; `:Assertion` / `:JoinPath`
        nodes are detach-deleted, transitively removing provenance edges.
        With ``preserve_assertions`` (resume builds), `:Assertion` nodes
        survive: they are the resume cache ``process_table`` reads to skip
        completed tables, and re-materialization rebuilds their provenance
        edges. Shared concept and physical nodes keep their content, but
        elements left with no relationships — orphaned `:Alias` and
        edge-less concept nodes — are then garbage-collected
        (`delete_orphaned_nodes`, J/H).
        """
        from sema.graph import loader_utils as _lu
        self._run(
            "MATCH ()-[r {source_schema: $schema}]-() DELETE r",
            schema=schema_name,
        )
        if not preserve_assertions:
            self._run(
                "MATCH (a:Assertion {source_schema: $schema}) "
                "DETACH DELETE a",
                schema=schema_name,
            )
        self._run(
            "MATCH (jp:JoinPath {source_schema: $schema}) "
            "DETACH DELETE jp",
            schema=schema_name,
        )
        _lu.delete_orphaned_nodes(self)

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
