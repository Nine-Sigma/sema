"""§1.5(a) resolved value-mapping store — DuckDB-backed decision cache.

Canonical home is **DuckDB** (the store is small). There is intentionally no
Databricks seed sync: the SQLGlot transform compiler (US-010) *inlines* the
resolved decisions into the staging write, so the warehouse never reads this
table. Grain is location-independent — one row per distinct source code per
target binding (no ``source_schema``/``source_table``) — so an
``OncoTree:LUAD → concept`` decision is reused across every study.

US-006 (the resolver) is the SOLE writer; downstream stories READ via
:meth:`ValueMappingStore.read_all` / :meth:`get`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import duckdb

from sema.models.planner.lifecycle import Status
from sema.resolve.value_mapping_store_utils import (
    GRAIN_KEY,
    ValueMapping,
    create_table_sql,
    from_row,
    select_all_sql,
    select_by_grain_sql,
    to_params,
    upsert_sql,
)

DEFAULT_SCHEMA = "sema_resolve"
DEFAULT_TABLE = "value_mapping"


class ValueMappingStore:
    """Thin DuckDB layer enforcing the §1.5(a) frozen schema and grain."""

    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        schema: str = DEFAULT_SCHEMA,
        table: str = DEFAULT_TABLE,
    ) -> None:
        self._conn = connection
        self._schema = schema
        self._table = table
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self._schema}"')
        self._conn.execute(create_table_sql(self._schema, self._table))

    def upsert(self, decisions: Iterable[ValueMapping]) -> int:
        sql = upsert_sql(self._schema, self._table)
        count = 0
        for decision in decisions:
            self._conn.execute(sql, to_params(decision))
            count += 1
        return count

    def read_all(self) -> list[ValueMapping]:
        rows = self._conn.execute(select_all_sql(self._schema, self._table)).fetchall()
        return [from_row(tuple(row), Status) for row in rows]

    def get(
        self,
        *,
        source_vocabulary: str,
        normalized_source_value: str,
        target_property_ref: str,
        resolver_policy_ref: str,
        vocab_release: str,
    ) -> ValueMapping | None:
        key = {
            "source_vocabulary": source_vocabulary,
            "normalized_source_value": normalized_source_value,
            "target_property_ref": target_property_ref,
            "resolver_policy_ref": resolver_policy_ref,
            "vocab_release": vocab_release,
        }
        params = [key[col] for col in GRAIN_KEY]
        row = self._conn.execute(
            select_by_grain_sql(self._schema, self._table), params
        ).fetchone()
        return from_row(tuple(row), Status) if row is not None else None

    def column_names(self) -> list[str]:
        # Scope to the store's own catalog: an attached second DuckDB database
        # (e.g. ~/.sema/poc.duckdb) may expose the same schema.table, which would
        # otherwise interleave a duplicate column list. The store's read/write
        # SQL is unqualified, so current_database() is the catalog it targets.
        rows = self._conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_catalog = current_database() "
            "AND table_schema = ? AND table_name = ? ORDER BY ordinal_position",
            [self._schema, self._table],
        ).fetchall()
        return [r[0] for r in rows]

    def count(self) -> int:
        row = self._conn.execute(
            f'SELECT COUNT(*) FROM "{self._schema}"."{self._table}"'
        ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ValueMappingStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def open_duckdb_value_mapping_store(
    db_path: str,
    *,
    schema: str = DEFAULT_SCHEMA,
    table: str = DEFAULT_TABLE,
) -> ValueMappingStore:
    """Open (creating parent dirs) a DuckDB-backed value-mapping store."""
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    return ValueMappingStore(conn, schema=schema, table=table)
