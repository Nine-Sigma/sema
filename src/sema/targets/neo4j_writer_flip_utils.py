"""Cypher flip statements for `Neo4jGraphWriter.flip_prior_generations`.

Each statement scopes the `is_current=false` flip to logical artifacts
touched by the current load AND `target_schema_snapshot_hash <>
$current_hash`. Lazy-load preservation: artifacts outside the loaded
subset are not flipped.
"""

from __future__ import annotations

from typing import Any

from sema.targets.materializer_ops import CurrentFlipOp


def flip_statements(op: CurrentFlipOp) -> list[tuple[str, dict[str, Any]]]:
    base = {
        "target_model_id": op.target_model_id,
        "target_model_version": op.target_model_version,
        "current_hash": op.current_snapshot_hash,
    }
    out: list[tuple[str, dict[str, Any]]] = []
    out.extend(_entity_flip(op, base))
    out.extend(_property_flip(op, base))
    out.extend(_obligation_flip(op, base))
    out.extend(_enrichment_flip(op, base))
    out.extend(_term_flip(op, base))
    out.extend(_constraint_flip(op, base))
    out.extend(_vocab_binding_flip(op, base))
    out.extend(_context_card_flip(op, base))
    return out


def _entity_flip(
    op: CurrentFlipOp, base: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    if not op.entity_qualified_names:
        return []
    return [
        (
            "MATCH (n:Entity) WHERE n.target_model_id = $target_model_id "
            "AND n.target_model_version = $target_model_version "
            "AND n.qualified_name IN $names "
            "AND n.target_schema_snapshot_hash <> $current_hash "
            "SET n.is_current = false",
            {**base, "names": list(op.entity_qualified_names)},
        )
    ]


def _property_flip(
    op: CurrentFlipOp, base: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    if not op.property_keys:
        return []
    return [
        (
            "UNWIND $keys AS key "
            "MATCH (n:Property) WHERE n.target_model_id = $target_model_id "
            "AND n.target_model_version = $target_model_version "
            "AND n.parent_entity_qualified_name = key[0] "
            "AND n.name = key[1] "
            "AND n.target_schema_snapshot_hash <> $current_hash "
            "SET n.is_current = false",
            {**base, "keys": [list(t) for t in op.property_keys]},
        )
    ]


def _obligation_flip(
    op: CurrentFlipOp, base: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    if not op.obligation_target_entities:
        return []
    return [
        (
            "MATCH (n:TargetObligation) WHERE n.target_model_id = $target_model_id "
            "AND n.target_model_version = $target_model_version "
            "AND n.target_entity IN $entities "
            "AND n.target_schema_snapshot_hash <> $current_hash "
            "SET n.is_current = false",
            {**base, "entities": list(op.obligation_target_entities)},
        )
    ]


def _enrichment_flip(
    op: CurrentFlipOp, base: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    if not op.enrichment_entity_refs:
        return []
    return [
        (
            "MATCH (n:EnrichmentDecision) WHERE n.target_model_id = $target_model_id "
            "AND n.target_model_version = $target_model_version "
            "AND n.entity_ref IN $names "
            "AND n.target_schema_snapshot_hash <> $current_hash "
            "SET n.is_current = false",
            {**base, "names": list(op.enrichment_entity_refs)},
        )
    ]


def _term_flip(
    op: CurrentFlipOp, base: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    if not op.term_keys:
        return []
    return [
        (
            "UNWIND $keys AS key "
            "MATCH (n:Term) WHERE n.target_model_id = $target_model_id "
            "AND n.target_model_version = $target_model_version "
            "AND n.vocabulary_name = key[0] AND n.code = key[1] "
            "AND n.target_schema_snapshot_hash <> $current_hash "
            "SET n.is_current = false",
            {**base, "keys": [list(t) for t in op.term_keys]},
        )
    ]


def _constraint_flip(
    op: CurrentFlipOp, base: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    if not op.property_keys:
        return []
    return [
        (
            "UNWIND $keys AS key "
            "MATCH (n:Constraint) WHERE n.target_model_id = $target_model_id "
            "AND n.target_model_version = $target_model_version "
            "AND n.attached_property_id = key[0] + '.' + key[1] "
            "AND n.target_schema_snapshot_hash <> $current_hash "
            "SET n.is_current = false",
            {**base, "keys": [list(t) for t in op.property_keys]},
        )
    ]


def _vocab_binding_flip(
    op: CurrentFlipOp, base: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    if not op.vocabulary_binding_keys:
        return []
    return [
        (
            "UNWIND $keys AS key "
            "MATCH (n:VocabularyBinding) WHERE n.target_model_id = $target_model_id "
            "AND n.target_model_version = $target_model_version "
            "AND n.parent_entity_qualified_name = key[0] "
            "AND n.property_name = key[1] AND n.vocabulary_name = key[2] "
            "AND n.target_schema_snapshot_hash <> $current_hash "
            "SET n.is_current = false",
            {**base, "keys": [list(t) for t in op.vocabulary_binding_keys]},
        )
    ]


def _context_card_flip(
    op: CurrentFlipOp, base: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    """Flip prior cards for the same entity that don't match the current
    (target_schema_snapshot_hash, card_version) tuple. Card-only bumps
    leave the schema hash unchanged, so flipping by snapshot hash alone
    would miss them; we flip by entity-vs-current-identity instead."""
    if not op.context_card_keys:
        return []
    return [
        (
            "UNWIND $keys AS key "
            "MATCH (n:ContextCard) WHERE n.target_model_id = $target_model_id "
            "AND n.target_model_version = $target_model_version "
            "AND n.entity_qualified_name = key[0] "
            "AND NOT (n.target_schema_snapshot_hash = $current_hash "
            "AND n.card_version = key[1]) "
            "SET n.is_current = false",
            {**base, "keys": [list(t) for t in op.context_card_keys]},
        )
    ]


__all__ = ["flip_statements"]
