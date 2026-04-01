from __future__ import annotations

from typing import Any

from sema.models.context import (
    AncestryTerm,
    GovernedValue,
    JoinPath,
    PhysicalAsset,
    Provenance,
    ResolvedEntity,
    ResolvedProperty,
    SemanticCandidateSet,
    SemanticContextObject,
)
from sema.pipeline.context_utils import (
    _apply_visibility_policy,
    _build_ancestry,
    _build_entities_and_assets,
    _build_governed_values,
    _build_join_paths,
    _build_metrics,
    _filter_entity_candidates,
)


def prune_to_sco(
    candidate_set: SemanticCandidateSet,
    consumer: str = "nl2sql",
    max_entities: int = 10,
    max_joins: int = 20,
) -> SemanticContextObject:
    """Prune candidate set into a task-ready SCO."""
    visible_candidates = _apply_visibility_policy(
        candidate_set.candidates, consumer
    )
    visible_set = SemanticCandidateSet(
        query=candidate_set.query, candidates=visible_candidates
    )
    entity_candidates = _filter_entity_candidates(visible_set, max_entities)
    entities, physical_assets = _build_entities_and_assets(entity_candidates)
    join_paths = _build_join_paths(visible_set, max_joins)
    governed_values = _build_governed_values(visible_set)
    ancestry = _build_ancestry(visible_set)
    metrics = _build_metrics(visible_set)

    return SemanticContextObject(
        entities=entities,
        metrics=metrics,
        physical_assets=physical_assets,
        join_paths=join_paths,
        governed_values=governed_values,
        ancestry=ancestry,
        consumer=consumer,
    )
