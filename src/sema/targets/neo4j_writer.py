"""Neo4j-backed `GraphWriter` for `TargetModelMaterializer`.

Takes a `neo4j.Driver` directly (no `sema.graph` import) so the
import-boundary rules stay intact: only `materializer.py` is allowed
to reach into `sema.graph`. The writer's contract is a sequence of
typed `WriteOp`s; this module turns them into Cypher MERGEs.
"""

from __future__ import annotations

from typing import Any

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
)
from sema.targets.neo4j_writer_utils import (
    constraint_merge,
    context_card_merge,
    enrichment_decision_merge,
    entity_merge,
    flip_statements,
    property_merge,
    relationship_merge,
    target_obligation_merge,
    term_merge,
    vocabulary_binding_merge,
)


class Neo4jGraphWriter:
    """`GraphWriter` that issues hash-versioned MERGEs against Neo4j.

    Each call opens a session and runs one Cypher statement. Callers
    that need a single transaction (e.g., to fail-atomically across
    multiple writes) should wrap a sequence of calls in their own
    `driver.session()` block and use `Neo4jGraphWriter.from_session`.
    """

    def __init__(self, driver: Any) -> None:
        self._driver = driver

    @classmethod
    def from_session(cls, session: Any) -> "Neo4jGraphWriter":
        instance = cls.__new__(cls)
        instance._driver = None
        instance._session = session  # type: ignore[attr-defined]
        return instance

    def _run(self, cypher: str, params: dict[str, Any]) -> None:
        if self._driver is None:
            self._session.run(cypher, **params)  # type: ignore[attr-defined]
            return
        with self._driver.session() as session:
            session.run(cypher, **params)

    def write_entity(self, op: EntityOp) -> None:
        self._run(*entity_merge(op))

    def write_property(self, op: PropertyOp) -> None:
        self._run(*property_merge(op))

    def write_term(self, op: TermOp) -> None:
        self._run(*term_merge(op))

    def write_constraint(self, op: object) -> None:
        if not isinstance(op, ConstraintOp):
            return
        self._run(*constraint_merge(op))

    def write_target_obligation(self, op: TargetObligationOp) -> None:
        self._run(*target_obligation_merge(op))

    def write_enrichment_decision(self, op: EnrichmentDecisionOp) -> None:
        self._run(*enrichment_decision_merge(op))

    def write_relationship(self, op: RelationshipOp) -> None:
        self._run(*relationship_merge(op))

    def write_vocabulary_binding(self, op: VocabularyBindingOp) -> None:
        self._run(*vocabulary_binding_merge(op))

    def write_context_card(self, op: ContextCardOp) -> None:
        self._run(*context_card_merge(op))

    def flip_prior_generations(self, op: CurrentFlipOp) -> None:
        for cypher, params in flip_statements(op):
            self._run(cypher, params)


__all__ = ["Neo4jGraphWriter"]
