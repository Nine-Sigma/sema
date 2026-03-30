"""Batch UNWIND helpers for GraphLoader.

Each function takes a ``loader`` (GraphLoader) as its first argument and
delegates to ``loader._run`` for Cypher execution.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader


def batch_upsert_entities(
    loader: GraphLoader, entities: list[dict[str, Any]],
) -> None:
    if not entities:
        return
    resolved_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {**e, "resolved_at": resolved_at, "id": str(uuid.uuid4())}
        for e in entities
    ]
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (e:Entity {datasource_id: r.datasource_id, "
        "table_key: r.table_key}) "
        "ON CREATE SET e.id = r.id "
        "SET e.name = r.name, "
        "e.description = r.description, e.source = r.source, "
        "e.confidence = r.confidence, "
        "e.status = 'ACTIVE', "
        "e.resolved_at = r.resolved_at "
        "WITH e, r "
        "MERGE (t:Table {name: r.table_name, "
        "schema_name: r.schema_name, catalog: r.catalog}) "
        "MERGE (e)-[:ENTITY_ON_TABLE]->(t)",
        rows=rows,
    )


def batch_upsert_properties(
    loader: GraphLoader, properties: list[dict[str, Any]],
) -> None:
    if not properties:
        return
    resolved_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {**p, "resolved_at": resolved_at, "id": str(uuid.uuid4())}
        for p in properties
    ]
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (p:Property {datasource_id: r.datasource_id, "
        "column_key: r.column_key}) "
        "ON CREATE SET p.id = r.id "
        "SET p.name = r.name, "
        "p.entity_name = r.entity_name, "
        "p.semantic_type = r.semantic_type, "
        "p.source = r.source, "
        "p.confidence = r.confidence, "
        "p.status = 'ACTIVE', "
        "p.resolved_at = r.resolved_at "
        "WITH p, r "
        "MERGE (e:Entity {datasource_id: r.datasource_id, "
        "table_key: r.table_key}) "
        "MERGE (e)-[:HAS_PROPERTY]->(p) "
        "WITH p, r "
        "MERGE (c:Column {name: r.column_name, "
        "table_name: r.table_name, "
        "schema_name: r.schema_name, catalog: r.catalog}) "
        "MERGE (p)-[:PROPERTY_ON_COLUMN]->(c)",
        rows=rows,
    )


def batch_upsert_terms(
    loader: GraphLoader, terms: list[dict[str, Any]],
) -> None:
    if not terms:
        return
    resolved_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {**t, "resolved_at": resolved_at, "id": str(uuid.uuid4())}
        for t in terms
    ]
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (t:Term {vocabulary_name: r.vocabulary_name, "
        "code: r.code}) "
        "ON CREATE SET t.id = r.id "
        "SET t.label = r.label, t.source = r.source, "
        "t.confidence = r.confidence, "
        "t.status = 'ACTIVE', "
        "t.resolved_at = r.resolved_at",
        rows=rows,
    )


def batch_upsert_aliases(
    loader: GraphLoader,
    aliases: list[dict[str, Any]],
    parent_label: str,
) -> None:
    if not aliases:
        return
    resolved_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {**a, "resolved_at": resolved_at, "id": str(uuid.uuid4())}
        for a in aliases
    ]
    loader._run(
        f"UNWIND $rows AS r "
        f"MERGE (a:Alias {{target_key: r.target_key, text: r.text}}) "
        f"ON CREATE SET a.id = r.id "
        f"SET a.source = r.source, a.confidence = r.confidence, "
        f"a.resolved_at = r.resolved_at, "
        f"a.status = 'ACTIVE', "
        f"a.is_preferred = r.is_preferred, "
        f"a.description = r.description "
        f"WITH a, r "
        f"MERGE (p{parent_label} {{name: r.parent_name}}) "
        f"MERGE (a)-[:REFERS_TO]->(p)",
        rows=rows,
    )


def batch_upsert_value_sets(
    loader: GraphLoader, value_sets: list[dict[str, Any]],
) -> None:
    if not value_sets:
        return
    rows = [
        {**vs, "id": str(uuid.uuid4())} for vs in value_sets
    ]
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (vs:ValueSet {datasource_id: r.datasource_id, "
        "column_key: r.column_key}) "
        "ON CREATE SET vs.id = r.id "
        "SET vs.name = r.name, "
        "vs.status = 'ACTIVE' "
        "WITH vs, r "
        "MERGE (c:Column {name: r.column_name, "
        "table_name: r.table_name, "
        "schema_name: r.schema_name, catalog: r.catalog}) "
        "MERGE (vs)-[:STORED_IN]->(c)",
        rows=rows,
    )


def batch_upsert_join_paths(
    loader: GraphLoader, join_paths: list[dict[str, Any]],
) -> None:
    if not join_paths:
        return
    resolved_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            **jp,
            "resolved_at": resolved_at,
            "id": str(uuid.uuid4()),
            "join_predicates_json": json.dumps(
                jp["join_predicates"]
            ),
        }
        for jp in join_paths
    ]
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (jp:JoinPath {datasource_id: r.datasource_id, "
        "from_table: r.from_table, to_table: r.to_table, "
        "join_columns_hash: r.join_columns_hash}) "
        "ON CREATE SET jp.id = r.id "
        "SET jp.name = r.name, "
        "jp.join_predicates = r.join_predicates_json, "
        "jp.hop_count = r.hop_count, "
        "jp.source = r.source, "
        "jp.confidence = r.confidence, "
        "jp.sql_snippet = r.sql_snippet, "
        "jp.cardinality_hint = r.cardinality_hint, "
        "jp.status = 'ACTIVE', "
        "jp.resolved_at = r.resolved_at",
        rows=rows,
    )


def batch_upsert_vocabularies(
    loader: GraphLoader, vocabularies: list[dict[str, Any]],
) -> None:
    """Upsert Vocabulary nodes. Merge key: {name} (global)."""
    if not vocabularies:
        return
    resolved_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {**v, "resolved_at": resolved_at, "id": str(uuid.uuid4())}
        for v in vocabularies
    ]
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (v:Vocabulary {name: r.name}) "
        "ON CREATE SET v.id = r.id "
        "SET v.status = 'ACTIVE', "
        "v.resolved_at = r.resolved_at",
        rows=rows,
    )


def batch_create_classified_as(
    loader: GraphLoader,
    edges: list[dict[str, Any]],
) -> None:
    """Create (Property)-[:CLASSIFIED_AS]->(Vocabulary) edges."""
    if not edges:
        return
    loader._run(
        "UNWIND $rows AS r "
        "MATCH (p:Property {datasource_id: r.datasource_id, "
        "column_key: r.column_key}) "
        "MERGE (v:Vocabulary {name: r.vocabulary_name}) "
        "MERGE (p)-[:CLASSIFIED_AS]->(v)",
        rows=edges,
    )


def batch_create_in_vocabulary(
    loader: GraphLoader,
    edges: list[dict[str, Any]],
) -> None:
    """Create (Term)-[:IN_VOCABULARY]->(Vocabulary) edges."""
    if not edges:
        return
    loader._run(
        "UNWIND $rows AS r "
        "MATCH (t:Term {vocabulary_name: r.vocabulary_name, "
        "code: r.code}) "
        "MERGE (v:Vocabulary {name: r.vocabulary_name}) "
        "MERGE (t)-[:IN_VOCABULARY]->(v)",
        rows=edges,
    )
