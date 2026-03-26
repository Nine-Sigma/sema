from __future__ import annotations

from typing import Any

from sema.models.context import (
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
    _build_entities_and_assets,
    _build_governed_values,
    _build_join_paths,
    _filter_entity_candidates,
)


def prune_to_sco(
    candidate_set: SemanticCandidateSet,
    consumer_hint: str = "nl2sql",
    max_entities: int = 10,
    max_joins: int = 20,
) -> SemanticContextObject:
    """Prune candidate set into a task-ready SCO."""
    entity_candidates = _filter_entity_candidates(candidate_set, max_entities)
    entities, physical_assets = _build_entities_and_assets(entity_candidates)
    join_paths = _build_join_paths(candidate_set, max_joins)
    governed_values = _build_governed_values(candidate_set)

    return SemanticContextObject(
        entities=entities,
        physical_assets=physical_assets,
        join_paths=join_paths,
        governed_values=governed_values,
        consumer_hint=consumer_hint,
    )
