from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sema.engine.vocabulary_utils import (
    build_synonym_assertions,
    detect_by_llm,
    detect_by_llm_client,
    detect_by_llm_legacy,
    expand_via_llm,
)
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
)

logger = logging.getLogger(__name__)

ICD10_PATTERN = re.compile(r"^[A-Z]\d{2}(\.\d{1,2})?$", re.IGNORECASE)
AJCC_STAGE_PATTERN = re.compile(
    r"^stage\s+(0|I{1,3}V?|IV)(A\d?|B\d?|C\d?)?$", re.IGNORECASE
)
TNM_PATTERN = re.compile(
    r"^T\d[a-c]?N\d[a-c]?M\d[a-c]?$", re.IGNORECASE
)


def detect_vocabulary_pattern(values: list[str]) -> dict[str, Any] | None:
    """Detect vocabulary from value patterns using regex."""
    if not values:
        return None

    for pattern, vocab_name in [
        (ICD10_PATTERN, "ICD-10"),
        (AJCC_STAGE_PATTERN, "AJCC Staging"),
        (TNM_PATTERN, "TNM Classification"),
    ]:
        match_count = sum(
            1 for v in values if pattern.match(str(v).strip())
        )
        match_ratio = match_count / len(values) if values else 0
        if match_ratio >= 0.6:
            return {
                "vocabulary": vocab_name,
                "confidence": min(
                    0.9 + (match_ratio - 0.6) * 0.25, 1.0
                ),
                "match_ratio": match_ratio,
            }

    return None


def infer_hierarchy(values: list[str]) -> list[tuple[str, str]]:
    """Infer parent-child relationships from value patterns."""
    if not values:
        return []

    sorted_values = sorted(set(values), key=len)
    hierarchy: list[tuple[str, str]] = []

    for i, parent in enumerate(sorted_values):
        for child in sorted_values[i + 1:]:
            if child.startswith(parent) and len(child) > len(parent):
                suffix = child[len(parent):]
                if re.match(
                    r"^[.A-Z0-9]{1,3}$", suffix, re.IGNORECASE
                ):
                    has_intermediate = any(
                        other != parent
                        and other != child
                        and child.startswith(other)
                        and other.startswith(parent)
                        and len(parent) < len(other) < len(child)
                        for other in sorted_values
                    )
                    if not has_intermediate:
                        hierarchy.append((parent, child))

    return hierarchy


class VocabularyEngine:
    """L3: Vocabulary detection, hierarchy inference, and synonym expansion."""

    def __init__(self, llm: Any = None, run_id: str | None = None, llm_client: Any = None) -> None:
        self._llm = llm
        self._llm_client = llm_client
        self._run_id = run_id or str(uuid.uuid4())

    def _make_assertion(
        self,
        subject_ref: str,
        predicate: AssertionPredicate,
        payload: dict[str, Any],
        source: str = "llm_interpretation",
        confidence: float = 0.75,
    ) -> Assertion:
        return Assertion(
            id=str(uuid.uuid4()),
            subject_ref=subject_ref,
            predicate=predicate,
            payload=payload,
            source=source,
            confidence=confidence,
            run_id=self._run_id,
            observed_at=datetime.now(timezone.utc),
        )

    def detect_vocabulary(
        self, column_ref: str, values: list[str]
    ) -> list[Assertion]:
        """Detect which vocabulary a column's values belong to."""
        assertions: list[Assertion] = []

        pattern_result = detect_vocabulary_pattern(values)
        if pattern_result:
            assertions.append(self._make_assertion(
                column_ref,
                AssertionPredicate.VOCABULARY_MATCH,
                {"value": pattern_result["vocabulary"]},
                source="pattern_match",
                confidence=pattern_result["confidence"],
            ))
            return assertions

        assertions.extend(self._detect_by_llm(column_ref, values))
        return assertions

    def _detect_by_llm(
        self, subject_ref: str, values: list[str]
    ) -> list[Assertion]:
        return detect_by_llm(self, subject_ref, values)

    def _detect_by_llm_client(
        self, subject_ref: str, values: list[str]
    ) -> list[Assertion]:
        return detect_by_llm_client(self, subject_ref, values)

    def _detect_by_llm_legacy(
        self, subject_ref: str, values: list[str]
    ) -> list[Assertion]:
        return detect_by_llm_legacy(self, subject_ref, values)

    def infer_value_hierarchy(
        self, column_ref: str, values: list[str]
    ) -> list[Assertion]:
        """Infer parent-child hierarchy from value patterns."""
        hierarchy = infer_hierarchy(values)
        assertions: list[Assertion] = []

        for parent, child in hierarchy:
            assertions.append(self._make_assertion(
                column_ref,
                AssertionPredicate.PARENT_OF,
                {"parent": parent, "child": child},
                source="pattern_match",
                confidence=0.8,
            ))

        return assertions

    def expand_synonyms(
        self, column_ref: str, terms: list[dict[str, str]]
    ) -> list[Assertion]:
        """Generate synonym assertions for decoded terms via LLM."""
        synonyms = self._expand_via_llm(column_ref, terms)
        return self._build_synonym_assertions(column_ref, synonyms)

    def _expand_via_llm(
        self, subject_ref: str, terms: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        return expand_via_llm(self, subject_ref, terms)

    def _build_synonym_assertions(
        self, subject_ref: str, synonyms: list[dict[str, Any]]
    ) -> list[Assertion]:
        return build_synonym_assertions(self, subject_ref, synonyms)

    def process_column(
        self,
        column_ref: str,
        values: list[str],
        decoded_values: list[dict[str, str]] | None = None,
    ) -> list[Assertion]:
        """Full L3 pipeline for a single column."""
        assertions: list[Assertion] = []

        assertions.extend(self.detect_vocabulary(column_ref, values))
        assertions.extend(self.infer_value_hierarchy(column_ref, values))
        if decoded_values:
            assertions.extend(
                self.expand_synonyms(column_ref, decoded_values)
            )

        return assertions
