"""Provenance SUBJECT/OBJECT edge materialization (US-006, bug-239).

Mechanism: a provenance assertion's subject/object endpoint is resolved from
its ``subject_ref`` / ``object_ref`` to the physical :Table or :Column node it
describes, via the canonical physical key (``CanonicalRef.parse``) — the same
key the physical nodes are MERGEd on. The earlier mechanism matched on
``Assertion.subject_id`` / ``object_id``, which are never populated anywhere in
``src/sema``; it was a permanent no-op (bug-239) and is deleted. Endpoints
whose ref cannot be parsed are skipped with a debug log, so no unguarded
null-MATCH round-trips remain.

Edges are written with batched UNWIND statements (one per endpoint role and
target label), never one Cypher round-trip per assertion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sema.log import logger
from sema.models.assertions import Assertion
from sema.models.physical_key import CanonicalRef

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader

PROVENANCE_PREDICATES = frozenset({
    "has_entity_name", "has_property_name", "has_alias",
    "has_semantic_type", "has_decoded_value",
    "vocabulary_match", "parent_of", "has_join_evidence",
})

_TABLE_EDGE = (
    "UNWIND $rows AS r "
    "MATCH (assertion:Assertion {{id: r.a_id}}) "
    "MATCH (t:Table {{name: r.name, schema_name: r.schema_name, "
    "catalog: r.catalog}}) "
    "MERGE (assertion)-[:{rel}]->(t)"
)

_COLUMN_EDGE = (
    "UNWIND $rows AS r "
    "MATCH (assertion:Assertion {{id: r.a_id}}) "
    "MATCH (c:Column {{name: r.name, table_name: r.table_name, "
    "schema_name: r.schema_name, catalog: r.catalog}}) "
    "MERGE (assertion)-[:{rel}]->(c)"
)


def _match_row(a_id: str, ref: str) -> tuple[str, dict[str, Any]] | None:
    """Resolve a ref to ('table'|'column', merge-key row), or None if unparseable."""
    try:
        pk = CanonicalRef.parse(ref)
    except ValueError:
        logger.debug(
            "Provenance: skipping unresolvable ref '{}' for assertion {}",
            ref, a_id,
        )
        return None
    schema_name = pk.schema or ""
    if pk.column:
        return "column", {
            "a_id": a_id, "name": pk.column, "table_name": pk.table,
            "schema_name": schema_name, "catalog": pk.catalog_or_db,
        }
    return "table", {
        "a_id": a_id, "name": pk.table,
        "schema_name": schema_name, "catalog": pk.catalog_or_db,
    }


def _collect_rows(
    assertions: list[Assertion], ref_attr: str,
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {"table": [], "column": []}
    for a in assertions:
        if a.predicate.value not in PROVENANCE_PREDICATES:
            continue
        ref = getattr(a, ref_attr)
        if not ref:
            continue
        resolved = _match_row(a.id, ref)
        if resolved is None:
            continue
        kind, row = resolved
        rows[kind].append(row)
    return rows


def _emit(
    loader: GraphLoader, rel: str,
    rows_by_kind: dict[str, list[dict[str, Any]]],
) -> None:
    if rows_by_kind["table"]:
        loader._run(_TABLE_EDGE.format(rel=rel), rows=rows_by_kind["table"])
    if rows_by_kind["column"]:
        loader._run(_COLUMN_EDGE.format(rel=rel), rows=rows_by_kind["column"])


def materialize_provenance_edges(
    loader: GraphLoader, assertions: list[Assertion],
) -> None:
    """Link each provenance assertion to the physical node it describes.

    SUBJECT edges resolve from ``subject_ref``; OBJECT edges from
    ``object_ref`` (only join/mapping predicates carry one).
    """
    _emit(loader, "SUBJECT", _collect_rows(assertions, "subject_ref"))
    _emit(loader, "OBJECT", _collect_rows(assertions, "object_ref"))
