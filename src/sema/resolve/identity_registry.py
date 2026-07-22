"""S1-01 — generic identity registry: DuckDB-backed canonical id assignment.

Mirrors :mod:`sema.resolve.value_mapping_store` (DuckDB-canonical, one writer,
downstream reads only) but implements a **two-level** identity map so Stage B can
revise identity without rewriting persisted PKs (D1):

    (source_namespace, source_entity_key)  # level-1 unique key
        -> source_entity_uid               # stable per-source-entity surrogate
        -> entity_id                        # registry-assigned canonical id

Stage A writes an identity map: each distinct source entity gets its own
``entity_id``. Assignment is an atomic get-or-create on the transactional unique
key (D7): a re-run reads the same assignment and never re-mints an existing one.

The module is DOMAIN-GENERIC (D6/R29) — it names no ``person``/OMOP literal; the
OMOP ``person`` binding is applied by the policy/compile layer that reads
``entity_id`` into ``person_id``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import duckdb

from sema.resolve.identity_registry_utils import (
    IdentityAssignment,
    create_table_sql,
    from_row,
    insert_ignore_sql,
    max_entity_id_sql,
    select_all_sql,
    select_by_grain_sql,
    source_entity_uid,
)

DEFAULT_SCHEMA = "sema_identity"
DEFAULT_TABLE = "entity_identity"

SourceKey = tuple[str, str]


class IdentityRegistry:
    """Thin DuckDB layer enforcing the two-level identity schema and grain."""

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

    def get_or_create(
        self, source_keys: Iterable[SourceKey], *, run_id: str
    ) -> dict[SourceKey, IdentityAssignment]:
        """Assign (or look up) a canonical ``entity_id`` per source key.

        Idempotent and replay-safe: keys already in the registry keep their
        assignment; new keys continue from ``MAX(entity_id)``. Returns every
        requested key mapped to its authoritative assignment.
        """
        pairs = self._dedup_validate(source_keys)
        existing = self._read_existing()
        new_pairs = [p for p in pairs if p not in existing]
        if new_pairs:
            self._insert_new(new_pairs, run_id)
            existing = self._read_existing()
        return {p: existing[p] for p in pairs}

    def _insert_new(self, new_pairs: list[SourceKey], run_id: str) -> None:
        base = self._max_entity_id()
        sql = insert_ignore_sql(self._schema, self._table)
        for offset, (namespace, key) in enumerate(new_pairs, start=1):
            row = IdentityAssignment(
                source_namespace=namespace,
                source_entity_key=key,
                source_entity_uid=source_entity_uid(namespace, key),
                entity_id=base + offset,
                run_id=run_id,
            )
            self._conn.execute(
                sql,
                [
                    row.source_namespace,
                    row.source_entity_key,
                    row.source_entity_uid,
                    row.entity_id,
                    row.run_id,
                ],
            )

    def get(
        self, source_namespace: str, source_entity_key: str
    ) -> IdentityAssignment | None:
        row = self._conn.execute(
            select_by_grain_sql(self._schema, self._table),
            [source_namespace, source_entity_key],
        ).fetchone()
        return from_row(tuple(row)) if row is not None else None

    def read_all(self) -> list[IdentityAssignment]:
        rows = self._conn.execute(select_all_sql(self._schema, self._table)).fetchall()
        return [from_row(tuple(row)) for row in rows]

    def column_names(self) -> list[str]:
        # Scope to the store's own catalog (see ValueMappingStore.column_names):
        # an attached second DuckDB (e.g. ~/.sema/poc.duckdb) may expose the same
        # schema.table, which would otherwise double the column list.
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

    def _read_existing(self) -> dict[SourceKey, IdentityAssignment]:
        return {
            (a.source_namespace, a.source_entity_key): a for a in self.read_all()
        }

    def _max_entity_id(self) -> int:
        row = self._conn.execute(
            max_entity_id_sql(self._schema, self._table)
        ).fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _dedup_validate(source_keys: Iterable[SourceKey]) -> list[SourceKey]:
        seen: set[SourceKey] = set()
        pairs: list[SourceKey] = []
        for namespace, key in source_keys:
            if not key:
                raise ValueError(
                    "source_entity_key must be non-empty; a blank source key "
                    "routes to NO_MAP upstream, never to a synthetic identity"
                )
            pair = (namespace, key)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
        return pairs

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> IdentityRegistry:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def open_duckdb_identity_registry(
    db_path: str,
    *,
    schema: str = DEFAULT_SCHEMA,
    table: str = DEFAULT_TABLE,
) -> IdentityRegistry:
    """Open (creating parent dirs) a DuckDB-backed identity registry."""
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    return IdentityRegistry(conn, schema=schema, table=table)
