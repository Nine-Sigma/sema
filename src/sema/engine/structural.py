from __future__ import annotations

import logging
from collections import defaultdict

from sema.engine.structural_utils import (
    process_columns,
    process_tables,
)
from sema.graph.loader import GraphLoader
from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.constants import parse_unity_ref_strict

logger = logging.getLogger(__name__)


class StructuralEngine:
    """L1: Deterministic parsing of extraction assertions into physical graph nodes."""

    def __init__(self, loader: GraphLoader) -> None:
        self._loader = loader

    def process(self, assertions: list[Assertion]) -> None:
        by_subject: dict[str, list[Assertion]] = defaultdict(list)
        for a in assertions:
            by_subject[a.subject_ref].append(a)

        created_catalogs: set[str] = set()
        created_schemas: set[tuple[str, str]] = set()

        process_tables(self._loader, by_subject, created_catalogs, created_schemas)
        process_columns(self._loader, by_subject)
