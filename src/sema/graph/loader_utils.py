"""Batch UNWIND helpers extracted from GraphLoader to keep loader.py under 400 lines.

Each function takes a ``loader`` (GraphLoader) as its first argument and
delegates to ``loader._run`` for Cypher execution.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader


def batch_upsert_entities(
    loader: GraphLoader, entities: list[dict[str, Any]]
) -> None:
    if not entities:
        return
    resolved_at = datetime.now(timezone.utc).isoformat()
    rows = [{**e, "resolved_at": resolved_at} for e in entities]
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (e:Entity {name: r.name}) "
        "SET e.description = r.description, e.source = r.source, "
        "e.confidence = r.confidence, e.resolved_at = r.resolved_at "
        "WITH e, r "
        "MERGE (t:Table {name: r.table_name, schema_name: r.schema_name, catalog: r.catalog}) "
        "MERGE (e)-[:IMPLEMENTED_BY]->(t)",
        rows=rows,
    )


def batch_upsert_properties(
    loader: GraphLoader, properties: list[dict[str, Any]]
) -> None:
    if not properties:
        return
    resolved_at = datetime.now(timezone.utc).isoformat()
    rows = [{**p, "resolved_at": resolved_at} for p in properties]
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (p:Property {name: r.name, entity_name: r.entity_name}) "
        "SET p.semantic_type = r.semantic_type, p.source = r.source, "
        "p.confidence = r.confidence, p.resolved_at = r.resolved_at "
        "WITH p, r "
        "MERGE (e:Entity {name: r.entity_name}) "
        "MERGE (e)-[:HAS_PROPERTY]->(p) "
        "WITH p, r "
        "MERGE (c:Column {name: r.column_name, table_name: r.table_name, "
        "schema_name: r.schema_name, catalog: r.catalog}) "
        "MERGE (p)-[:IMPLEMENTED_BY]->(c)",
        rows=rows,
    )


def batch_upsert_terms(
    loader: GraphLoader, terms: list[dict[str, Any]]
) -> None:
    if not terms:
        return
    resolved_at = datetime.now(timezone.utc).isoformat()
    rows = [{**t, "resolved_at": resolved_at} for t in terms]
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (t:Term {code: r.code}) "
        "SET t.label = r.label, t.source = r.source, "
        "t.confidence = r.confidence, t.resolved_at = r.resolved_at",
        rows=rows,
    )


def batch_upsert_synonyms(
    loader: GraphLoader,
    synonyms: list[dict[str, Any]],
    parent_label: str,
) -> None:
    if not synonyms:
        return
    resolved_at = datetime.now(timezone.utc).isoformat()
    rows = [{**s, "resolved_at": resolved_at} for s in synonyms]
    loader._run(
        f"UNWIND $rows AS r "
        f"MERGE (s:Synonym {{text: r.text}}) "
        f"SET s.source = r.source, s.confidence = r.confidence, "
        f"s.resolved_at = r.resolved_at "
        f"WITH s, r "
        f"MERGE (p{parent_label} {{name: r.parent_name}}) "
        f"MERGE (s)-[:SYNONYM_OF]->(p)",
        rows=rows,
    )


def batch_upsert_value_sets(
    loader: GraphLoader, value_sets: list[dict[str, Any]]
) -> None:
    if not value_sets:
        return
    loader._run(
        "UNWIND $rows AS r "
        "MERGE (vs:ValueSet {name: r.name}) "
        "WITH vs, r "
        "MERGE (c:Column {name: r.column_name, table_name: r.table_name, "
        "schema_name: r.schema_name, catalog: r.catalog}) "
        "MERGE (vs)-[:STORED_IN]->(c)",
        rows=value_sets,
    )
