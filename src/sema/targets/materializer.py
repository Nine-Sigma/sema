"""TargetModelMaterializer + GraphWriter Protocol + InMemoryGraphWriter.

The materializer is the only `sema.targets` module permitted to import
from `sema.graph` (per the import-boundary rule). This module must not
import from `sema.engine` or `sema.pipeline`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sema.models.target.enrichment import EnrichmentDecisionRecord
from sema.models.target.normalized import NormalizedTargetModel
from sema.targets.exceptions import (
    EnrichmentStatusDivergenceError,
    LoaderStageOrderError,
)
from sema.models.target.context_card import LoadedContextCard
from sema.targets.materializer_ops import (
    ConstraintOp,
    ContextCardOp,
    CurrentFlipOp,
    EnrichmentDecisionOp,
    EntityOp,
    PropertyOp,
    RelationshipOp,
    TargetObligationOp,
    TermOp,
    VocabularyBindingOp,
    WriteOp,
)
from sema.targets.materializer_utils import (
    build_current_flip_op,
    write_constraints,
    write_context_cards,
    write_decision_node,
    write_entity_and_properties,
    write_obligation,
    write_term,
    write_vocabulary_bindings,
)


@runtime_checkable
class GraphWriter(Protocol):
    def write_entity(self, op: EntityOp) -> None: ...

    def write_property(self, op: PropertyOp) -> None: ...

    def write_term(self, op: TermOp) -> None: ...

    def write_constraint(self, op: object) -> None: ...

    def write_target_obligation(self, op: TargetObligationOp) -> None: ...

    def write_enrichment_decision(self, op: EnrichmentDecisionOp) -> None: ...

    def write_relationship(self, op: RelationshipOp) -> None: ...

    def write_vocabulary_binding(self, op: VocabularyBindingOp) -> None: ...

    def write_context_card(self, op: ContextCardOp) -> None: ...

    def flip_prior_generations(self, op: CurrentFlipOp) -> None: ...


class InMemoryGraphWriter:
    """Test double that records every write call as a typed `WriteOp`."""

    def __init__(self) -> None:
        self.ops: list[WriteOp] = []

    def write_entity(self, op: EntityOp) -> None:
        self.ops.append(op)

    def write_property(self, op: PropertyOp) -> None:
        self.ops.append(op)

    def write_term(self, op: TermOp) -> None:
        self.ops.append(op)

    def write_constraint(self, op: object) -> None:
        if isinstance(op, ConstraintOp):
            self.ops.append(op)

    def write_target_obligation(self, op: TargetObligationOp) -> None:
        self.ops.append(op)

    def write_enrichment_decision(self, op: EnrichmentDecisionOp) -> None:
        self.ops.append(op)

    def write_relationship(self, op: RelationshipOp) -> None:
        self.ops.append(op)

    def write_vocabulary_binding(self, op: VocabularyBindingOp) -> None:
        self.ops.append(op)

    def write_context_card(self, op: ContextCardOp) -> None:
        self.ops.append(op)

    def flip_prior_generations(self, op: CurrentFlipOp) -> None:
        self.ops.append(op)


class StageGuard:
    """Tracks loader stage progression; materializer requires HASHED state."""

    NORMALIZED = "normalized"
    HASHED = "hashed"
    MATERIALIZED = "materialized"

    def __init__(self) -> None:
        self.state: str | None = None

    def transition_to(self, state: str) -> None:
        order = (self.NORMALIZED, self.HASHED, self.MATERIALIZED)
        prior_index = -1 if self.state is None else order.index(self.state)
        new_index = order.index(state)
        if new_index != prior_index + 1:
            raise LoaderStageOrderError(
                f"stage out of order: tried to enter {state!r} from {self.state!r}"
            )
        self.state = state

    def require_at_least(self, state: str) -> None:
        order = (self.NORMALIZED, self.HASHED, self.MATERIALIZED)
        current = -1 if self.state is None else order.index(self.state)
        target = order.index(state)
        if current < target:
            raise LoaderStageOrderError(
                f"materializer requires stage >= {state!r}; current={self.state!r}"
            )


class TargetModelMaterializer:
    @staticmethod
    def write(
        model: NormalizedTargetModel,
        target_schema_snapshot_hash: str,
        writer: GraphWriter,
        enrichment_decisions: list[EnrichmentDecisionRecord],
        cards_with_hash: list[LoadedContextCard] | None = None,
        stage_guard: StageGuard | None = None,
    ) -> list[EnrichmentDecisionRecord]:
        if stage_guard is not None:
            stage_guard.require_at_least(StageGuard.HASHED)
        descriptor = model.descriptor
        decisions_by_entity = {
            r.entity_ref.qualified_name: r for r in enrichment_decisions
        }
        obligations_by_entity = {o.target_entity: o for o in model.obligations}
        for entity in model.entities:
            decision = decisions_by_entity.get(entity.ref.qualified_name)
            if decision is None:
                raise EnrichmentStatusDivergenceError(
                    f"entity {entity.ref.qualified_name!r} has no EnrichmentDecisionRecord"
                )
            write_entity_and_properties(
                writer, descriptor, entity, target_schema_snapshot_hash, decision
            )
            write_constraints(
                writer,
                descriptor,
                entity,
                obligations_by_entity.get(entity.ref.qualified_name),
                target_schema_snapshot_hash,
            )
            write_decision_node(
                writer, descriptor, entity, target_schema_snapshot_hash, decision
            )
        for obligation in model.obligations:
            write_obligation(writer, descriptor, obligation, target_schema_snapshot_hash)
        for term in model.terms:
            write_term(writer, descriptor, term, target_schema_snapshot_hash)
        write_vocabulary_bindings(
            writer, descriptor, model.vocabulary_bindings, target_schema_snapshot_hash
        )
        write_context_cards(
            writer, descriptor, cards_with_hash or [], target_schema_snapshot_hash
        )
        writer.flip_prior_generations(
            build_current_flip_op(
                descriptor,
                model,
                target_schema_snapshot_hash,
                bindings=model.vocabulary_bindings,
                cards=cards_with_hash or [],
            )
        )
        if stage_guard is not None:
            stage_guard.transition_to(StageGuard.MATERIALIZED)
        return enrichment_decisions


__all__ = [
    "GraphWriter",
    "InMemoryGraphWriter",
    "TargetModelMaterializer",
    "StageGuard",
]
