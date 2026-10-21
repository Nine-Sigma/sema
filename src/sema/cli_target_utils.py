"""Helpers for `cli_target.py`: CLI output shapes, the projection binding,
and the recording driver behind `sema target project --writer in-memory`."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sema.graph.target_retrieval_loader import (
    EntityTableBinding,
    PhysicalSchemaBinding,
)
from sema.models.target.loaded import LoadedTarget


def build_summary(loaded: LoadedTarget) -> dict[str, Any]:
    return {
        "target_model_id": loaded.descriptor.target_model_id,
        "target_model_version": loaded.descriptor.target_model_version,
        "target_schema_snapshot_hash": loaded.target_schema_snapshot_hash,
        "aggregate_context_card_version": loaded.aggregate_context_card_version,
        "materialized_at": loaded.materialized_at.isoformat(),
        "entities": [
            {"qualified_name": e.qualified_name, "kind": e.kind.value}
            for e in loaded.entity_refs
        ],
        "enrichment_decisions": [
            _decision_to_dict(d) for d in loaded.enrichment_decisions
        ],
        "context_cards": [
            {
                "entity_ref": c.entity_ref.qualified_name,
                "card_version": c.card_version,
                "card_hash": c.card_hash,
            }
            for c in loaded.context_cards
        ],
    }


def _decision_to_dict(record: Any) -> dict[str, Any]:
    return {
        "entity_ref": record.entity_ref.qualified_name,
        "decisions": {
            facet.value: {
                "status": fd.status.value,
                "reason": fd.reason,
            }
            for facet, fd in record.decisions.items()
        },
    }


class ProjectBinding(BaseModel):
    """Operator-authored placement of target entities in one catalog.schema."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    catalog: str = Field(min_length=1)
    schema_name: str = Field(alias="schema", min_length=1)
    entity_tables: dict[str, str]

    def to_physical(self) -> PhysicalSchemaBinding:
        return PhysicalSchemaBinding(
            catalog=self.catalog,
            schema=self.schema_name,
            entity_tables=tuple(
                EntityTableBinding(entity, table)
                for entity, table in self.entity_tables.items()
            ),
        )


def load_project_binding(path: Path) -> PhysicalSchemaBinding:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ProjectBinding.model_validate(raw).to_physical()


class RecordingDriver:
    """Neo4j driver stand-in that records each statement instead of running it."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, dict[str, Any]]] = []

    def session(self) -> _RecordingSession:
        return _RecordingSession(self.statements)

    def close(self) -> None:
        return None


class _RecordingSession:
    def __init__(self, statements: list[tuple[str, dict[str, Any]]]) -> None:
        self._statements = statements

    def __enter__(self) -> _RecordingSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self._statements.append((cypher, params))
        return []


__all__ = ["ProjectBinding", "RecordingDriver", "build_summary", "load_project_binding"]
