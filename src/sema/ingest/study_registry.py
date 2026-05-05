from __future__ import annotations

from sema.ingest.duckdb_staging import Staging

_REGISTRY_SCHEMA = "_sema"
_REGISTRY_TABLE = "_sema_study_registry"
_QUALIFIED = f'"{_REGISTRY_SCHEMA}"."{_REGISTRY_TABLE}"'


class StudyCollisionError(RuntimeError):
    """Raised when two original study IDs sanitize to the same schema name."""


class StudyRegistry:
    def __init__(self, staging: Staging) -> None:
        self._staging = staging
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._staging.execute(f'CREATE SCHEMA IF NOT EXISTS "{_REGISTRY_SCHEMA}"')
        self._staging.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_QUALIFIED} (
                schema_name TEXT PRIMARY KEY,
                original_study_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def register(self, schema_name: str, original_study_id: str, source_type: str) -> None:
        existing = self._lookup(schema_name)
        if existing is not None:
            if existing == original_study_id:
                return
            raise StudyCollisionError(
                f"Schema name {schema_name!r} is already registered to study "
                f"{existing!r}; cannot also register study {original_study_id!r}. "
                f"Rename one of the studies or choose disambiguating IDs."
            )
        self._staging.execute(
            f"INSERT INTO {_QUALIFIED} (schema_name, original_study_id, source_type) "
            "VALUES (?, ?, ?)",
            [schema_name, original_study_id, source_type],
        )

    def _lookup(self, schema_name: str) -> str | None:
        row = self._staging.execute(
            f"SELECT original_study_id FROM {_QUALIFIED} WHERE schema_name = ?",
            [schema_name],
        ).fetchone()
        return row[0] if row else None

    def list_schemas(self) -> list[str]:
        rows = self._staging.execute(
            f"SELECT schema_name FROM {_QUALIFIED} ORDER BY schema_name"
        ).fetchall()
        return [r[0] for r in rows]

    def find_schema_for_study(self, study_id: str) -> str | None:
        row = self._staging.execute(
            f"SELECT schema_name FROM {_QUALIFIED} WHERE original_study_id = ?",
            [study_id],
        ).fetchone()
        return row[0] if row else None
