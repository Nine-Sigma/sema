"""Materializer helpers for VocabularyBinding + ContextCard writes.

Kept separate from `materializer_utils.py` so each helper file stays
under the 400-line cap.
"""

from __future__ import annotations

from typing import Any, Protocol

from sema.models.target.context_card import LoadedContextCard
from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.targets.materializer_ops import (
    ContextCardOp,
    RelationshipOp,
    VocabularyBindingOp,
)


class _WriterLike(Protocol):
    def write_vocabulary_binding(self, op: VocabularyBindingOp) -> None: ...
    def write_context_card(self, op: ContextCardOp) -> None: ...
    def write_relationship(self, op: RelationshipOp) -> None: ...


def write_vocabulary_bindings(
    writer: _WriterLike,
    descriptor: Any,
    bindings: list[VocabularyBindingDecl],
    snapshot_hash: str,
) -> None:
    for b in bindings:
        writer.write_vocabulary_binding(_binding_op(descriptor, b, snapshot_hash))
        writer.write_relationship(_binding_rel(descriptor, b, snapshot_hash))


def _binding_op(
    descriptor: Any,
    b: VocabularyBindingDecl,
    snapshot_hash: str,
) -> VocabularyBindingOp:
    return VocabularyBindingOp(
        target_model_id=descriptor.target_model_id,
        target_model_version=descriptor.target_model_version,
        target_schema_snapshot_hash=snapshot_hash,
        parent_entity_qualified_name=b.entity_ref.qualified_name,
        property_name=b.property_name,
        vocabulary_name=b.vocabulary.name,
        vocabulary_source=b.vocabulary.source.value,
        domain=b.domain,
        require_standard=b.require_standard,
        allow_zero_default=b.allow_zero_default,
        effective_date_ref=b.effective_date_ref,
        resolver_policy_ref=b.resolver_policy_ref,
    )


def _binding_rel(
    descriptor: Any,
    b: VocabularyBindingDecl,
    snapshot_hash: str,
) -> RelationshipOp:
    versioned = _versioned_keys(descriptor, snapshot_hash)
    return RelationshipOp(
        rel_type="HAS_VOCABULARY_BINDING",
        target_schema_snapshot_hash=snapshot_hash,
        from_label="Property",
        from_keys={
            **versioned,
            "parent_entity_qualified_name": b.entity_ref.qualified_name,
            "name": b.property_name,
        },
        to_label="VocabularyBinding",
        to_keys={
            **versioned,
            "parent_entity_qualified_name": b.entity_ref.qualified_name,
            "property_name": b.property_name,
            "vocabulary_name": b.vocabulary.name,
        },
    )


def write_context_cards(
    writer: _WriterLike,
    descriptor: Any,
    cards: list[LoadedContextCard],
    snapshot_hash: str,
) -> None:
    for card in cards:
        writer.write_context_card(_card_op(descriptor, card, snapshot_hash))
        writer.write_relationship(_card_rel(descriptor, card, snapshot_hash))


def _card_op(
    descriptor: Any, card: LoadedContextCard, snapshot_hash: str
) -> ContextCardOp:
    return ContextCardOp(
        target_model_id=descriptor.target_model_id,
        target_model_version=descriptor.target_model_version,
        target_schema_snapshot_hash=snapshot_hash,
        entity_qualified_name=card.entity_ref.qualified_name,
        card_version=card.card_version,
        card_hash=card.card_hash,
        description=card.description,
        examples=list(card.examples),
        obligation_summary=card.obligation_summary,
        curated_synonyms=list(card.curated_synonyms),
    )


def _card_rel(
    descriptor: Any, card: LoadedContextCard, snapshot_hash: str
) -> RelationshipOp:
    versioned = _versioned_keys(descriptor, snapshot_hash)
    return RelationshipOp(
        rel_type="HAS_CONTEXT_CARD",
        target_schema_snapshot_hash=snapshot_hash,
        from_label="Entity",
        from_keys={
            **versioned,
            "qualified_name": card.entity_ref.qualified_name,
        },
        to_label="ContextCard",
        to_keys={
            **versioned,
            "entity_qualified_name": card.entity_ref.qualified_name,
            "card_version": card.card_version,
        },
    )


def _versioned_keys(descriptor: Any, snapshot_hash: str) -> dict[str, str]:
    return {
        "target_model_id": descriptor.target_model_id,
        "target_model_version": descriptor.target_model_version,
        "target_schema_snapshot_hash": snapshot_hash,
    }


__all__ = ["write_vocabulary_bindings", "write_context_cards"]
