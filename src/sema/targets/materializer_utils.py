"""Helpers for `TargetModelMaterializer`.

Keeps `materializer.py` thin (it must stay under the 400-line cap and
remains the only `sema.targets` module permitted to import `sema.graph`,
so non-graph helpers live here).
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

from sema.models.planner.target_model import DomainConstraint, ForeignKeyObligation
from sema.models.target.context_card import LoadedContextCard
from sema.models.target.enrichment import EnrichmentDecisionRecord
from sema.models.target.entity import TargetEntityDecl
from sema.models.target.normalized import NormalizedTargetModel
from sema.models.target.obligation import TargetObligationDecl
from sema.models.target.properties import PropertyKind, TargetPropertyDecl
from sema.models.target.term import TargetTermDecl
from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.targets.hashing_utils import canonical_dumps
from sema.targets.materializer_ops import (
    ConstraintOp,
    CurrentFlipOp,
    EnrichmentDecisionOp,
    EntityOp,
    PropertyOp,
    RelationshipOp,
    TargetObligationOp,
    TermOp,
)


class _WriterLike(Protocol):
    def write_entity(self, op: EntityOp) -> None: ...
    def write_property(self, op: PropertyOp) -> None: ...
    def write_term(self, op: TermOp) -> None: ...
    def write_constraint(self, op: object) -> None: ...
    def write_target_obligation(self, op: TargetObligationOp) -> None: ...
    def write_enrichment_decision(self, op: EnrichmentDecisionOp) -> None: ...
    def write_relationship(self, op: RelationshipOp) -> None: ...


def constraints_for_obligation(
    descriptor: Any,
    entity: TargetEntityDecl,
    obligation: TargetObligationDecl,
    snapshot_hash: str,
) -> list[ConstraintOp]:
    columnar = {
        p.name for p in entity.properties if p.property_kind is PropertyKind.COLUMN
    }
    return [
        _domain_constraint_op(descriptor, entity, dc, snapshot_hash)
        for dc in obligation.domain_constraints
        if dc.property_name in columnar
    ]


def _domain_constraint_op(
    descriptor: Any,
    entity: TargetEntityDecl,
    dc: DomainConstraint,
    snapshot_hash: str,
) -> ConstraintOp:
    payload = {"domain_id": dc.domain_id}
    payload_hash = _payload_hash(payload)
    return ConstraintOp(
        target_model_id=descriptor.target_model_id,
        target_model_version=descriptor.target_model_version,
        target_schema_snapshot_hash=snapshot_hash,
        attached_property_id=f"{entity.ref.qualified_name}.{dc.property_name}",
        constraint_kind="domain_binding",
        payload=payload,
        payload_hash=payload_hash,
    )


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()


def write_entity_and_properties(
    writer: _WriterLike,
    descriptor: Any,
    entity: TargetEntityDecl,
    snapshot_hash: str,
    decision: EnrichmentDecisionRecord,
) -> None:
    enrichment_status = {
        facet.value: fd.status.value for facet, fd in decision.decisions.items()
    }
    writer.write_entity(
        EntityOp(
            target_model_id=descriptor.target_model_id,
            target_model_version=descriptor.target_model_version,
            target_schema_snapshot_hash=snapshot_hash,
            qualified_name=entity.ref.qualified_name,
            kind=entity.ref.kind.value,
            enrichment_status=enrichment_status,
        )
    )
    for prop in entity.properties:
        writer.write_property(_property_op(descriptor, entity, prop, snapshot_hash))
        writer.write_relationship(
            _has_property_rel(descriptor, entity, prop, snapshot_hash)
        )


def _property_op(
    descriptor: Any,
    entity: TargetEntityDecl,
    prop: TargetPropertyDecl,
    snapshot_hash: str,
) -> PropertyOp:
    is_endpoint = prop.property_kind is PropertyKind.ENDPOINT
    return PropertyOp(
        target_model_id=descriptor.target_model_id,
        target_model_version=descriptor.target_model_version,
        target_schema_snapshot_hash=snapshot_hash,
        parent_entity_qualified_name=entity.ref.qualified_name,
        name=prop.name,
        type=prop.type,
        nullable=prop.nullable,
        synonyms=list(prop.synonyms),
        decoded_values=dict(prop.decoded_values),
        property_kind=prop.property_kind.value,
        endpoint_role=prop.endpoint_role,
        endpoint_target_entity_qualified_name=prop.endpoint_target_entity_qualified_name,
        endpoint_cardinality=prop.endpoint_cardinality,
        endpoint_nullable=prop.endpoint_nullable,
        materialized_as_edge_property=False if is_endpoint else prop.materialized_as_edge_property,
    )


def _has_property_rel(
    descriptor: Any,
    entity: TargetEntityDecl,
    prop: TargetPropertyDecl,
    snapshot_hash: str,
) -> RelationshipOp:
    return RelationshipOp(
        rel_type="HAS_PROPERTY",
        target_schema_snapshot_hash=snapshot_hash,
        from_label="Entity",
        from_keys={
            "target_model_id": descriptor.target_model_id,
            "target_model_version": descriptor.target_model_version,
            "target_schema_snapshot_hash": snapshot_hash,
            "qualified_name": entity.ref.qualified_name,
        },
        to_label="Property",
        to_keys={
            "target_model_id": descriptor.target_model_id,
            "target_model_version": descriptor.target_model_version,
            "target_schema_snapshot_hash": snapshot_hash,
            "parent_entity_qualified_name": entity.ref.qualified_name,
            "name": prop.name,
        },
    )


def write_decision_node(
    writer: _WriterLike,
    descriptor: Any,
    entity: TargetEntityDecl,
    snapshot_hash: str,
    decision: EnrichmentDecisionRecord,
) -> None:
    decisions_json = canonical_dumps(
        {
            facet.value: {
                "status": fd.status.value,
                "reason": fd.reason,
                "decided_at": fd.decided_at.isoformat(),
            }
            for facet, fd in decision.decisions.items()
        }
    )
    writer.write_enrichment_decision(
        EnrichmentDecisionOp(
            target_model_id=descriptor.target_model_id,
            target_model_version=descriptor.target_model_version,
            target_schema_snapshot_hash=snapshot_hash,
            entity_ref=entity.ref.qualified_name,
            decisions_json=decisions_json,
            decided_at=decision.decided_at,
        )
    )
    writer.write_relationship(
        RelationshipOp(
            rel_type="HAS_ENRICHMENT_DECISION",
            target_schema_snapshot_hash=snapshot_hash,
            from_label="Entity",
            from_keys=_entity_keys(descriptor, entity.ref.qualified_name, snapshot_hash),
            to_label="EnrichmentDecision",
            to_keys={
                **_versioned_keys(descriptor, snapshot_hash),
                "entity_ref": entity.ref.qualified_name,
            },
        )
    )


def write_obligation(
    writer: _WriterLike,
    descriptor: Any,
    obligation: TargetObligationDecl,
    snapshot_hash: str,
) -> None:
    payload = obligation.model_dump(mode="json")
    payload["foreign_keys"] = [_fk_payload(fk) for fk in obligation.foreign_keys]
    writer.write_target_obligation(
        TargetObligationOp(
            target_model_id=descriptor.target_model_id,
            target_model_version=descriptor.target_model_version,
            target_schema_snapshot_hash=snapshot_hash,
            target_entity=obligation.target_entity,
            payload=payload,
        )
    )
    writer.write_relationship(
        RelationshipOp(
            rel_type="HAS_OBLIGATION",
            target_schema_snapshot_hash=snapshot_hash,
            from_label="Entity",
            from_keys=_entity_keys(descriptor, obligation.target_entity, snapshot_hash),
            to_label="TargetObligation",
            to_keys={
                **_versioned_keys(descriptor, snapshot_hash),
                "target_entity": obligation.target_entity,
            },
        )
    )


def _fk_payload(fk: ForeignKeyObligation) -> dict[str, object]:
    return fk.model_dump(mode="json")


def write_term(
    writer: _WriterLike,
    descriptor: Any,
    term: TargetTermDecl,
    snapshot_hash: str,
) -> None:
    writer.write_term(
        TermOp(
            target_model_id=descriptor.target_model_id,
            target_model_version=descriptor.target_model_version,
            target_schema_snapshot_hash=snapshot_hash,
            vocabulary_name=term.vocabulary.name,
            code=term.code,
            display=term.display,
        )
    )


def write_constraints(
    writer: _WriterLike,
    descriptor: Any,
    entity: TargetEntityDecl,
    obligation: TargetObligationDecl | None,
    snapshot_hash: str,
) -> None:
    if obligation is None:
        return
    for op in constraints_for_obligation(descriptor, entity, obligation, snapshot_hash):
        writer.write_constraint(op)


from sema.targets.materializer_binding_card_utils import (
    write_context_cards,
    write_vocabulary_bindings,
)


def build_current_flip_op(
    descriptor: Any,
    model: NormalizedTargetModel,
    snapshot_hash: str,
    bindings: list[VocabularyBindingDecl] | None = None,
    cards: list[LoadedContextCard] | None = None,
) -> CurrentFlipOp:
    entity_names = tuple(e.ref.qualified_name for e in model.entities)
    property_keys = tuple(
        (e.ref.qualified_name, p.name) for e in model.entities for p in e.properties
    )
    obligation_targets = tuple(o.target_entity for o in model.obligations)
    term_keys = tuple((t.vocabulary.name, t.code) for t in model.terms)
    binding_keys = tuple(
        (b.entity_ref.qualified_name, b.property_name, b.vocabulary.name)
        for b in (bindings or [])
    )
    card_keys = tuple(
        (c.entity_ref.qualified_name, c.card_version) for c in (cards or [])
    )
    return CurrentFlipOp(
        target_model_id=descriptor.target_model_id,
        target_model_version=descriptor.target_model_version,
        current_snapshot_hash=snapshot_hash,
        entity_qualified_names=entity_names,
        property_keys=property_keys,
        obligation_target_entities=obligation_targets,
        enrichment_entity_refs=entity_names,
        vocabulary_binding_keys=binding_keys,
        context_card_keys=card_keys,
        term_keys=term_keys,
    )


def _versioned_keys(descriptor: Any, snapshot_hash: str) -> dict[str, str]:
    return {
        "target_model_id": descriptor.target_model_id,
        "target_model_version": descriptor.target_model_version,
        "target_schema_snapshot_hash": snapshot_hash,
    }


def _entity_keys(
    descriptor: Any, qualified_name: str, snapshot_hash: str
) -> dict[str, str]:
    return {
        **_versioned_keys(descriptor, snapshot_hash),
        "qualified_name": qualified_name,
    }


__all__ = [
    "build_current_flip_op",
    "constraints_for_obligation",
    "write_constraints",
    "write_context_cards",
    "write_decision_node",
    "write_entity_and_properties",
    "write_obligation",
    "write_term",
    "write_vocabulary_bindings",
]
