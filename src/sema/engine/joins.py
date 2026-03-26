from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from itertools import combinations
from typing import Any

from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
)
from sema.models.constants import GENERIC_JOIN_COLUMNS

logger = logging.getLogger(__name__)


class JoinInferenceEngine:
    """Infer candidate joins between tables using heuristics."""

    def __init__(self, run_id: str | None = None) -> None:
        self._run_id = run_id or str(uuid.uuid4())

    def find_heuristic_joins(
        self, table_columns: dict[str, list[str]]
    ) -> list[dict[str, Any]]:
        """Find candidate joins by shared column names across tables."""
        candidates: list[dict[str, Any]] = []
        table_refs = list(table_columns.keys())

        for ref_a, ref_b in combinations(table_refs, 2):
            cols_a = set(table_columns[ref_a])
            cols_b = set(table_columns[ref_b])
            shared = cols_a & cols_b

            for col in shared:
                confidence = 0.7
                if col.lower() in GENERIC_JOIN_COLUMNS:
                    confidence = 0.4
                elif col.lower().endswith("_id"):
                    confidence = 0.8

                candidates.append({
                    "from_ref": ref_a,
                    "to_ref": ref_b,
                    "on_column": col,
                    "confidence": confidence,
                })

        return candidates

    def infer_joins(
        self, table_columns: dict[str, list[str]]
    ) -> list[Assertion]:
        """Produce join assertions from heuristic candidates."""
        candidates = self.find_heuristic_joins(table_columns)
        assertions: list[Assertion] = []

        for candidate in candidates:
            assertions.append(Assertion(
                id=str(uuid.uuid4()),
                subject_ref=candidate["from_ref"],
                predicate=AssertionPredicate.JOINS_TO,
                payload={
                    "on_column": candidate["on_column"],
                    "cardinality": "unknown",
                },
                object_ref=candidate["to_ref"],
                source="heuristic",
                confidence=candidate["confidence"],
                run_id=self._run_id,
                observed_at=datetime.now(timezone.utc),
            ))

        return assertions
