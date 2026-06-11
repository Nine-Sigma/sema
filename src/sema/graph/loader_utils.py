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

from sema.graph.term_identity_utils import term_namespace
from sema.models.extraction import ExtractedColumn

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader


UNSET_CONFIDENCE = -1.0


def confidence_wins(incoming: float, stored: float | None) -> bool:
    """Incoming write overwrites a shared node only if strictly more confident.

    Equal confidence keeps the existing value, so the final state of a node
    shared across studies is independent of build/write order (finding C).
    """
    return incoming > (stored if stored is not None else UNSET_CONFIDENCE)


def _confidence_guard(alias: str) -> str:
    """Cypher boolean: incoming confidence strictly beats the stored value."""
    return f"r.confidence > coalesce({alias}.confidence, {UNSET_CONFIDENCE})"


def _guarded_set(alias: str, fields: list[str]) -> str:
    """Build SET assignments that only apply when `win` is true."""
    return ", ".join(
        f"{alias}.{f} = CASE WHEN win THEN r.{f} ELSE {alias}.{f} END"
        for f in fields
    )


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


def batch_upsert_entities(
    loader: GraphLoader,
    entities: list[dict[str, Any]],
    source_schema: str | None = None,
) -> None:
    if not entities:
        return
    rows = _annotate_rows(entities, source_schema)
    guarded = _guarded_set(
        "e", ["description", "source", "confidence", "resolved_at"],
    )
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (e:Entity {name: r.name}) "
        "ON CREATE SET e.id = r.id "
        f"WITH e, r, {_confidence_guard('e')} AS win "
        f"SET {guarded}, "
        "e.status = 'ACTIVE', "
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
    rows = _annotate_rows(properties, source_schema)
    guarded = _guarded_set(
        "p", ["semantic_type", "source", "confidence", "resolved_at"],
    )
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (p:Property {entity_name: r.entity_name, name: r.name}) "
        "ON CREATE SET p.id = r.id "
        f"WITH p, r, {_confidence_guard('p')} AS win "
        f"SET {guarded}, "
        "p.status = 'ACTIVE', "
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
            "vocabulary_name": term_namespace(t.get("vocabulary_name")),
            "resolved_at": resolved_at,
            "id": str(uuid.uuid4()),
            "source_schema": source_schema,
        }
        for t in terms
    ]
    guarded = _guarded_set(
        "t", ["label", "source", "confidence", "resolved_at"],
    )
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (t:Term {vocabulary_name: r.vocabulary_name, code: r.code}) "
        "ON CREATE SET t.id = r.id "
        f"WITH t, r, {_confidence_guard('t')} AS win "
        f"SET {guarded}, "
        "t.status = 'ACTIVE', "
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
    """Create (Property)-[:CLASSIFIED_AS]->(Vocabulary) edges.

    ``source_schema`` is in the MERGE key so two studies asserting the same
    Property->Vocabulary association produce two edges, each independently
    deletable by ``delete_study_scoped`` (finding K).
    """
    if not edges:
        return
    loader._run(
        "UNWIND $rows AS r "
        "MATCH (p:Property {entity_name: r.entity_name, name: r.name}) "
        "MERGE (v:Vocabulary {name: r.vocabulary_name}) "
        "MERGE (p)-[:CLASSIFIED_AS "
        "{source_schema: r.source_schema}]->(v)",
        rows=edges,
    )


def batch_create_in_vocabulary(
    loader: GraphLoader,
    edges: list[dict[str, Any]],
) -> None:
    """Create (Term)-[:IN_VOCABULARY]->(Vocabulary) edges.

    ``source_schema`` is in the MERGE key so two studies asserting the same
    Term->Vocabulary association produce two edges, each independently
    deletable by ``delete_study_scoped`` (finding K).
    """
    if not edges:
        return
    loader._run(
        "UNWIND $rows AS r "
        "MATCH (t:Term {vocabulary_name: r.vocabulary_name, code: r.code}) "
        "MERGE (v:Vocabulary {name: r.vocabulary_name}) "
        "MERGE (t)-[:IN_VOCABULARY "
        "{source_schema: r.source_schema}]->(v)",
        rows=edges,
    )


_ORPHAN_CONCEPT_LABELS = ("Entity", "Property", "Term", "Vocabulary", "ValueSet")


def delete_orphaned_nodes(loader: GraphLoader) -> None:
    """Garbage-collect graph elements stranded by a study deletion.

    Run AFTER ``delete_study_scoped`` removes a study's edges. An ``:Alias``
    keeps ``source_schema`` only on its ``REFERS_TO`` edge (finding J), so the
    node itself survives the edge sweep — delete it once it has no remaining
    ``REFERS_TO``. Concept nodes are shared and intentionally never study-scoped
    (finding H); delete one only when it has no relationships at all, so any
    node still referenced by a surviving study is untouched. Aliases are dropped
    first because that can free their target concept node to be swept too.
    """
    loader._run(
        "MATCH (a:Alias) WHERE NOT (a)-[:REFERS_TO]->() DELETE a",
    )
    for label in _ORPHAN_CONCEPT_LABELS:
        loader._run(
            f"MATCH (n:{label}) WHERE NOT (n)--() DELETE n",
        )
