"""The key read form on a loaded relation (contract F6): `Table.key_columns`, an ordered list
of column names, and `Table.key_verification`. Both are absent when the relation has no
accepted key, so a rebuild clears a key the data no longer accepts."""
from __future__ import annotations

from collections.abc import Sequence

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
