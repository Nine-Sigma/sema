"""Batch UNWIND helpers for GraphLoader.

Each function takes a ``loader`` (GraphLoader) as its first argument and
delegates to ``loader._run`` for Cypher execution.

Study-derived edges (`:ENTITY_ON_TABLE`, `:HAS_PROPERTY`,
`:PROPERTY_ON_COLUMN`, `:HAS_VALUE_SET`, `:REFERS_TO`) include
`source_schema` IN THE MERGE MATCH KEY so two studies emitting the same
logical edge produce two distinct relationships, each independently
scoped-deletable.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sema.models.extraction import ExtractedColumn

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader


_FETCH_COLUMNS_QUERY = (
    "MATCH (c:Column) WHERE c.schema_name = $schema_name "
    "RETURN c.name AS name, c.table_name AS table_name, "
    "c.catalog AS catalog, c.schema_name AS schema_name, "
    "c.data_type AS data_type, c.nullable AS nullable, "
    "c.comment AS comment"
)


def fetch_columns_by_schema(
    loader: GraphLoader, schema_name: str,
) -> list[ExtractedColumn]:
    rows = loader._run_read(
        _FETCH_COLUMNS_QUERY, schema_name=schema_name,
    )
    return [
        ExtractedColumn(
            name=row["name"],
            table_name=row["table_name"],
            catalog=row.get("catalog", ""),
            schema=row["schema_name"],
            data_type=row.get("data_type", "UNKNOWN"),
            nullable=bool(row.get("nullable", True)),
            comment=row.get("comment"),
        )
        for row in rows
    ]


def _annotate_rows(
    items: list[dict[str, Any]],
    source_schema: str | None,
) -> list[dict[str, Any]]:
    resolved_at = datetime.now(timezone.utc).isoformat()
    return [
        {
            **item,
            "resolved_at": resolved_at,
            "id": str(uuid.uuid4()),
            "source_schema": source_schema,
        }
        for item in items
    ]


def _require_source_schema(method: str, source_schema: str | None) -> None:
    if source_schema is None:
        raise ValueError(
            f"{method} requires source_schema; the relationship MERGE key "
            f"includes source_schema for multi-study isolation"
        )


def batch_upsert_entities(
    loader: GraphLoader,
    entities: list[dict[str, Any]],
    source_schema: str | None = None,
) -> None:
    if not entities:
        return
    _require_source_schema("batch_upsert_entities", source_schema)
    rows = _annotate_rows(entities, source_schema)
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (e:Entity {name: r.name}) "
        "ON CREATE SET e.id = r.id "
        "SET e.description = r.description, e.source = r.source, "
        "e.confidence = r.confidence, "
        "e.status = 'ACTIVE', "
        "e.resolved_at = r.resolved_at, "
        "e.model_role = coalesce(e.model_role, 'SOURCE'), "
        "e.source_id = coalesce(e.source_id, r.source_schema, r.source) "
        "WITH e, r "
        "MERGE (t:Table {name: r.table_name, "
        "schema_name: r.schema_name, catalog: r.catalog}) "
        "MERGE (e)-[link:ENTITY_ON_TABLE "
        "{source_schema: r.source_schema}]->(t)",
        rows=rows,
    )


def batch_upsert_properties(
    loader: GraphLoader,
    properties: list[dict[str, Any]],
    source_schema: str | None = None,
) -> None:
    if not properties:
        return
    _require_source_schema("batch_upsert_properties", source_schema)
    rows = _annotate_rows(properties, source_schema)
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (p:Property {entity_name: r.entity_name, name: r.name}) "
        "ON CREATE SET p.id = r.id "
        "SET p.semantic_type = r.semantic_type, "
        "p.source = r.source, "
        "p.confidence = r.confidence, "
        "p.status = 'ACTIVE', "
        "p.resolved_at = r.resolved_at, "
        "p.model_role = coalesce(p.model_role, 'SOURCE'), "
        "p.source_id = coalesce(p.source_id, r.source_schema, r.source) "
        "WITH p, r "
        "MERGE (e:Entity {name: r.entity_name}) "
        "SET e.model_role = coalesce(e.model_role, 'SOURCE'), "
        "e.source_id = coalesce(e.source_id, r.source_schema, r.source) "
        "MERGE (e)-[hp:HAS_PROPERTY "
        "{source_schema: r.source_schema}]->(p) "
        "WITH p, r "
        "MERGE (c:Column {name: r.column_name, "
        "table_name: r.table_name, "
        "schema_name: r.schema_name, catalog: r.catalog}) "
        "MERGE (p)-[poc:PROPERTY_ON_COLUMN "
        "{source_schema: r.source_schema}]->(c)",
        rows=rows,
    )


def batch_upsert_terms(
    loader: GraphLoader, terms: list[dict[str, Any]],
    source_schema: str | None = None,
) -> None:
    if not terms:
        return
    resolved_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            **t,
            "resolved_at": resolved_at,
            "id": str(uuid.uuid4()),
            "source_schema": source_schema,
        }
        for t in terms
    ]
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (t:Term {code: r.code}) "
        "ON CREATE SET t.id = r.id "
        "SET t.label = r.label, t.source = r.source, "
        "t.confidence = r.confidence, "
        "t.vocabulary_name = r.vocabulary_name, "
        "t.status = 'ACTIVE', "
        "t.resolved_at = r.resolved_at, "
        "t.model_role = coalesce(t.model_role, 'SOURCE'), "
        "t.source_id = coalesce(t.source_id, r.source_schema, r.source)",
        rows=rows,
    )


def batch_upsert_aliases(
    loader: GraphLoader,
    aliases: list[dict[str, Any]],
    parent_label: str,
    source_schema: str | None = None,
) -> None:
    if not aliases:
        return
    _require_source_schema("batch_upsert_aliases", source_schema)
    rows = _annotate_rows(aliases, source_schema)
    if parent_label == ":Property":
        parent_match = (
            "MERGE (p:Property {entity_name: r.parent_entity_name, "
            "name: r.parent_name})"
        )
    else:
        parent_match = f"MERGE (p{parent_label} {{name: r.parent_name}})"
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (a:Alias {target_key: r.target_key, text: r.text}) "
        "ON CREATE SET a.id = r.id "
        "SET a.source = r.source, a.confidence = r.confidence, "
        "a.resolved_at = r.resolved_at, "
        "a.status = 'ACTIVE', "
        "a.is_preferred = r.is_preferred, "
        "a.description = r.description "
        "WITH a, r "
        f"{parent_match} "
        "MERGE (a)-[ref:REFERS_TO "
        "{source_schema: r.source_schema}]->(p)",
        rows=rows,
    )


def batch_upsert_value_sets(
    loader: GraphLoader,
    value_sets: list[dict[str, Any]],
    source_schema: str | None = None,
) -> None:
    if not value_sets:
        return
    _require_source_schema("batch_upsert_value_sets", source_schema)
    rows = _annotate_rows(value_sets, source_schema)
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (vs:ValueSet {column_ref: r.column_ref}) "
        "ON CREATE SET vs.id = r.id "
        "SET vs.name = r.name, "
        "vs.status = 'ACTIVE' "
        "WITH vs, r "
        "MERGE (c:Column {name: r.column_name, "
        "table_name: r.table_name, "
        "schema_name: r.schema_name, catalog: r.catalog}) "
        "MERGE (c)-[hvs:HAS_VALUE_SET "
        "{source_schema: r.source_schema}]->(vs)",
        rows=rows,
    )


def batch_upsert_join_paths(
    loader: GraphLoader,
    join_paths: list[dict[str, Any]],
    source_schema: str | None = None,
) -> None:
    if not join_paths:
        return
    _require_source_schema("batch_upsert_join_paths", source_schema)
    resolved_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            **jp,
            "resolved_at": resolved_at,
            "id": str(uuid.uuid4()),
            "source_schema": source_schema,
            "join_predicates_json": json.dumps(
                jp["join_predicates"]
            ),
        }
        for jp in join_paths
    ]
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (jp:JoinPath {name: r.name, "
        "source_schema: r.source_schema}) "
        "ON CREATE SET jp.id = r.id "
        "SET jp.join_predicates = r.join_predicates_json, "
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
        "MATCH (p:Property {entity_name: r.entity_name, name: r.name}) "
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
        "MATCH (t:Term {code: r.code}) "
        "MERGE (v:Vocabulary {name: r.vocabulary_name}) "
        "MERGE (t)-[:IN_VOCABULARY]->(v)",
        rows=edges,
    )
