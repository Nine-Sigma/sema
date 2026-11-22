"""The key read form on a loaded relation (contract F6): `Table.key_columns`, an ordered list
of column names, and `Table.key_verification`. Both are absent when the relation has no
accepted key, so a rebuild clears a key the data no longer accepts."""
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from sema.graph.loader import GraphLoader

# (catalog, schema, table, (key columns, verification) or None for no accepted key)
TableKey = tuple[str, str, str, tuple[list[str], str] | None]

_SET_KEYS = (
    "UNWIND $keys AS k "
    "MATCH (t:Table {name: k.table, schema_name: k.schema, catalog: k.catalog}) "
    "SET t.key_columns = k.columns, t.key_verification = k.verification"
)


def write_table_keys(loader: GraphLoader, keys: Sequence[TableKey]) -> None:
    """Set each relation's key, or remove it; setting a property to null removes it."""
    if not keys:
        return
    loader._run(_SET_KEYS, keys=[
        {"catalog": catalog, "schema": schema, "table": table,
         "columns": key[0] if key else None, "verification": key[1] if key else None}
        for catalog, schema, table, key in keys
    ])


# The semantic stage's entity name and grain per table reference; a parameter named apart from
# the column read's, so a stand-in graph that dispatches on it answers nothing here.
_ENTITY_NAMES = (
    "MATCH (a:Assertion {predicate: 'has_entity_name'}) "
    "WHERE a.subject_ref IN $table_refs AND coalesce(a.status, '') <> 'superseded' "
    "RETURN a.subject_ref AS ref, a.payload AS payload ORDER BY a.observed_at"
)

# An edge the step creates gets its owner flag; one that existed keeps its own, unless this
# relation owns the entity. A property set to null is removed, so absent means the simple case.
_SET_ENTITY_EDGES = (
    "UNWIND $edges AS x "
    "MATCH (t:Table {name: x.table, schema_name: x.schema, catalog: x.catalog}) "
    "MERGE (e:Entity {name: x.entity}) "
    "ON CREATE SET e.id = randomUUID(), e.source = $source, e.status = 'ACTIVE', "
    "e.model_role = 'SOURCE', e.source_id = x.schema "
    "MERGE (e)-[r:ENTITY_ON_TABLE]->(t) "
    "ON CREATE SET r.owner = x.owner, r.source_schema = x.schema "
    "SET r.key_columns = x.key_columns, r.verification = x.verification, "
    "r.identity = x.identity, r.null_key_rows = x.null_key_rows, "
    "r.owner = CASE WHEN x.owner THEN true ELSE r.owner END"
)


def read_entity_names(
    loader: GraphLoader, table_refs: Sequence[str],
) -> dict[str, tuple[str | None, str | None]]:
    """Each table's latest entity name and grain hypothesis from the semantic stage."""
    found: dict[str, tuple[str | None, str | None]] = {}
    for row in loader._run_read(_ENTITY_NAMES, table_refs=list(table_refs)):
        payload = row["payload"]
        held = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
        found[row["ref"]] = (held.get("value"), held.get("grain"))
    return found


def write_entity_edges(
    loader: GraphLoader, edges: Sequence[dict[str, Any]], source: str,
) -> None:
    """One `ENTITY_ON_TABLE` edge per relation an entity is on (contract F6); each row holds
    catalog, schema, table, entity, owner, key_columns, verification, identity (a JSON string
    or None) and null_key_rows (None unless above 0); `source` marks an entity it creates."""
    if edges:
        loader._run(_SET_ENTITY_EDGES, edges=list(edges), source=source)
