"""Helpers for `cli_target.py`: shape `LoadedTarget` for CLI output."""

from __future__ import annotations

from typing import Any

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


__all__ = ["build_summary"]
