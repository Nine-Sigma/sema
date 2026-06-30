"""Vocabulary store query layer over the OMOP concept tables (US-003).

A thin :class:`VocabStore` reads ``concept`` / ``concept_relationship`` /
``concept_synonym`` (the names arrive via :class:`VocabStoreSchema`) and renders
each query per dialect so the *same* authored SQL runs on DuckDB (dev,
``~/.sema/poc.duckdb``) and Databricks (prod, ``workspace.vocabulary_omop``).
The 10M-concept vocabulary stays in the warehouse — never Neo4j.

The store is vocabulary-agnostic: the standardizing relationship name
(``'Maps to'``) and standard flag (``'S'``) are supplied by the caller from the
US-004 :class:`~sema.resolve.policy.ResolverPolicy`, not hardcoded here.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Sequence

from sema.resolve.vocab_store_utils import (
    ConceptRow,
    VocabStoreSchema,
    concept_by_code_query,
    concept_domain_query,
    maps_to_targets_query,
    row_to_concept,
)


class _Result(Protocol):
    def fetchall(self) -> list[Any]: ...


class _Connection(Protocol):
    def execute(self, sql: str, parameters: Sequence[Any] | None = ...) -> Any: ...


class VocabStoreBackend(str, Enum):
    DUCKDB = "duckdb"
    DATABRICKS = "databricks"


_NAMESPACE_BY_BACKEND = {
    VocabStoreBackend.DUCKDB: "vocabulary_omop",
    VocabStoreBackend.DATABRICKS: "workspace.vocabulary_omop",
}


class VocabStore:
    """Dialect-rendering query layer over a concept-vocabulary backend."""

    def __init__(
        self,
        connection: _Connection,
        *,
        schema: VocabStoreSchema,
        namespace: str,
        dialect: str = "duckdb",
    ) -> None:
        self._conn = connection
        self._schema = schema
        self._namespace = namespace
        self._dialect = dialect

    def concept_by_code(self, vocabulary: str, code: str) -> ConceptRow | None:
        sql = concept_by_code_query(self._schema, self._namespace).sql(
            dialect=self._dialect
        )
        rows = self._fetch(sql, [vocabulary, code])
        return row_to_concept(rows[0]) if rows else None

    def maps_to_targets(
        self,
        concept_id: str,
        *,
        relationship_id: str,
        standard_flag: str | None = None,
        only_valid: bool = False,
    ) -> list[ConceptRow]:
        ast = maps_to_targets_query(
            self._schema,
            self._namespace,
            standard=standard_flag is not None,
            only_valid=only_valid,
        )
        params: list[Any] = [concept_id, relationship_id]
        if standard_flag is not None:
            params.append(standard_flag)
        rows = self._fetch(ast.sql(dialect=self._dialect), params)
        return [row_to_concept(r) for r in rows]

    def concept_domain(self, concept_id: str) -> str | None:
        sql = concept_domain_query(self._schema, self._namespace).sql(
            dialect=self._dialect
        )
        rows = self._fetch(sql, [concept_id])
        if not rows:
            return None
        value = rows[0][0]
        return None if value is None else str(value)

    def _fetch(self, sql: str, params: Sequence[Any]) -> list[Any]:
        result = self._conn.execute(sql, params)
        if result is not None and hasattr(result, "fetchall"):
            return list(result.fetchall())
        return list(self._conn.fetchall())  # type: ignore[attr-defined]

    def close(self) -> None:
        close = getattr(self._conn, "close", None)
        if callable(close):
            close()


def open_duckdb_vocab_store(
    db_path: str,
    *,
    schema: VocabStoreSchema,
    namespace: str = "vocabulary_omop",
    read_only: bool = True,
) -> VocabStore:
    """Open a DuckDB-backed store at ``db_path`` (must exist)."""
    import duckdb

    path = Path(db_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {path}")
    conn = duckdb.connect(str(path), read_only=read_only)
    return VocabStore(conn, schema=schema, namespace=namespace, dialect="duckdb")


def namespace_for_backend(backend: VocabStoreBackend) -> str:
    """Default table namespace for a backend (DuckDB vs Databricks)."""
    return _NAMESPACE_BY_BACKEND[backend]
