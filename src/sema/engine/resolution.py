from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from sema.engine.resolution_utils import (
    _pick_winner,
    resolve_decoded_values,
    resolve_entities,
    resolve_hierarchies,
    resolve_properties,
    resolve_synonyms,
)
from sema.graph.loader import GraphLoader
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)
from sema.models.constants import parse_unity_ref, source_precedence

logger = logging.getLogger(__name__)


class ResolutionEngine:
    """Resolves assertions into canonical semantic graph nodes."""

    def __init__(self, loader: GraphLoader) -> None:
        self._loader = loader

    def resolve(self, assertions: list[Assertion]) -> None:
        for a in assertions:
            self._loader.store_assertion(a)

        groups: dict[tuple[str, str], list[Assertion]] = defaultdict(list)
        for a in assertions:
            groups[(a.subject_ref, a.predicate.value)].append(a)

        loader = self._loader
        resolve_entities(groups, loader)
        resolve_properties(groups, loader)
        resolve_decoded_values(groups, loader)
        resolve_synonyms(groups, loader)
        resolve_hierarchies(groups, loader)
