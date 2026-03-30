from __future__ import annotations

import logging
from collections import defaultdict

from sema.engine.structural_utils import (
    process_columns,
    process_tables,
)
from sema.graph.loader import GraphLoader
from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.constants import translate_ref

logger = logging.getLogger(__name__)


class StructuralEngine:
    """L1: Deterministic parsing of extraction assertions into physical graph nodes.

    Accepts refs in either legacy unity:// format or the canonical
    databricks://<workspace>/<catalog>/<schema>/<table>[/<column>] format.
    Pass workspace so that legacy refs can be translated on ingestion.
    """

    def __init__(self, loader: GraphLoader, workspace: str = "") -> None:
        self._loader = loader
        self._workspace = workspace

    def _normalize_ref(self, ref: str) -> str:
        """Translate legacy unity:// refs to databricks:// format when workspace is set."""
        if self._workspace and ref.startswith("unity://"):
            return translate_ref(ref, self._workspace)
        return ref

    def process(self, assertions: list[Assertion]) -> None:
        normalized: list[Assertion] = []
        for a in assertions:
            if self._workspace and (
                a.subject_ref.startswith("unity://")
                or (a.object_ref and a.object_ref.startswith("unity://"))
            ):
                a = a.model_copy(update={
                    "subject_ref": self._normalize_ref(a.subject_ref),
                    "object_ref": (
                        self._normalize_ref(a.object_ref) if a.object_ref else None
                    ),
                })
            normalized.append(a)

        by_subject: dict[str, list[Assertion]] = defaultdict(list)
        for a in normalized:
            by_subject[a.subject_ref].append(a)

        created_catalogs: set[str] = set()
        created_schemas: set[tuple[str, str]] = set()

        process_tables(self._loader, by_subject, created_catalogs, created_schemas)
        process_columns(self._loader, by_subject)
