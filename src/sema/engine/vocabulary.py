from __future__ import annotations

import logging
import re
import uuid
from typing import Final
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sema.engine.vocab_pattern import VocabPattern, detect_first_match
from sema.engine.vocabulary_utils import (
    apply_agreement_boost,
    build_synonym_assertions,
    detect_by_llm,
    expand_via_llm,
    should_infer_hierarchy,
)
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
)


@dataclass(frozen=True)
class VocabColumnContext:
    """L2-derived context passed to L3 for a single column."""

    column_name: str | None = None
    table_name: str | None = None
    entity_name: str | None = None
    semantic_type: str | None = None
    property_name: str | None = None
    vocabulary_guess: str | None = None
    vocabulary_guess_confidence: float = 0.0

logger = logging.getLogger(__name__)

VOCABULARY_PATTERNS: Final[list[VocabPattern]] = [
    # --- Healthcare ---
    VocabPattern(
        "ICD-10",
        re.compile(r"^[A-TV-Z]\d[\dA-Z](\.[\dA-Z]{1,4})?$", re.IGNORECASE),
        domain_affinity=frozenset({"healthcare"}),
    ),
    VocabPattern(
        "AJCC Staging",
        re.compile(r"^stage\s+(0|I{1,3}V?|IV)(A\d?|B\d?|C\d?)?$", re.IGNORECASE),
        domain_affinity=frozenset({"healthcare"}),
    ),
    VocabPattern(
        "TNM Classification",
        re.compile(r"^T\d[a-c]?N\d[a-c]?M\d[a-c]?$", re.IGNORECASE),
        domain_affinity=frozenset({"healthcare"}),
    ),
    VocabPattern(
        "LOINC",
        re.compile(r"^\d{1,5}-\d$"),
        domain_affinity=frozenset({"healthcare"}),
    ),
    VocabPattern(
        "NDC",
        re.compile(
            r"^(?:\d{4}-\d{4}-\d{2}|\d{5}-\d{3}-\d{2}"
            r"|\d{5}-\d{4}-\d{1}|\d{5}-\d{4}-\d{2})$"
        ),
        domain_affinity=frozenset({"healthcare"}),
    ),
    VocabPattern(
        "HGNC",
        re.compile(r"^HGNC:\d+$", re.IGNORECASE),
        domain_affinity=frozenset({"healthcare"}),
    ),
    VocabPattern(
        "CPT",
        re.compile(r"^\d{5}$"),
        context_keywords=frozenset({
            "cpt", "procedure", "proc", "service",
            "billing", "claim", "surgery",
        }),
        domain_affinity=frozenset({"healthcare"}),
    ),
    # --- Financial ---
    VocabPattern(
        "NAICS",
        re.compile(r"^\d{2,6}$"),
        context_keywords=frozenset({
            "naics", "industry", "sector", "classification",
        }),
        domain_affinity=frozenset({"financial"}),
    ),
    VocabPattern(
        "SIC",
        re.compile(r"^\d{4}$"),
        context_keywords=frozenset({
            "sic", "industry", "sector", "standard",
        }),
        domain_affinity=frozenset({"financial"}),
    ),
    VocabPattern(
        "CUSIP",
        re.compile(r"^[A-Z0-9]{9}$", re.IGNORECASE),
        context_keywords=frozenset({
            "cusip", "security", "bond", "equity", "isin",
        }),
        domain_affinity=frozenset({"financial"}),
    ),
    VocabPattern(
        "ISIN",
        re.compile(r"^[A-Z]{2}[A-Z0-9]{10}$", re.IGNORECASE),
        context_keywords=frozenset({
            "isin", "security", "international",
        }),
        domain_affinity=frozenset({"financial"}),
    ),
    VocabPattern(
        "SWIFT/BIC",
        re.compile(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$"),
        context_keywords=frozenset({
            "swift", "bic", "bank", "routing",
        }),
        domain_affinity=frozenset({"financial"}),
    ),
    VocabPattern(
        "LEI",
        re.compile(r"^[A-Z0-9]{20}$", re.IGNORECASE),
        context_keywords=frozenset({
            "lei", "entity", "legal", "identifier",
        }),
        domain_affinity=frozenset({"financial"}),
    ),
    # --- Geographic ---
    VocabPattern(
        "ZIP",
        re.compile(r"^\d{5}(-\d{4})?$"),
        context_keywords=frozenset({
            "zip", "postal", "address", "location", "city",
            "state", "mailing",
        }),
        domain_affinity=frozenset({
            "real_estate", "logistics", "general",
        }),
    ),
    VocabPattern(
        "FIPS",
        re.compile(r"^\d{2,5}$"),
        context_keywords=frozenset({
            "fips", "county", "state", "geographic",
        }),
        domain_affinity=frozenset({
            "real_estate", "logistics", "general",
        }),
    ),
    VocabPattern(
        "ISO-3166",
        re.compile(r"^[A-Z]{2,3}$"),
        context_keywords=frozenset({
            "country", "nation", "iso", "territory",
        }),
    ),
    # --- General ---
    VocabPattern(
        "ISO-4217",
        re.compile(r"^[A-Z]{3}$"),
        context_keywords=frozenset({
            "currency", "ccy", "money", "fx",
        }),
    ),
]


def detect_vocabulary_pattern(
    values: list[str],
    context: VocabColumnContext | None = None,
) -> dict[str, Any] | None:
    """Detect vocabulary from value patterns using regex."""
    return detect_first_match(VOCABULARY_PATTERNS, values, context)


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
        self,
        column_ref: str,
        values: list[str],
        context: VocabColumnContext | None = None,
    ) -> list[Assertion]:
        """Detect which vocabulary a column's values belong to."""
        pattern_result = detect_vocabulary_pattern(values, context)
        if pattern_result:
            return [self._make_assertion(
                column_ref,
                AssertionPredicate.VOCABULARY_MATCH,
                {"value": pattern_result["vocabulary"]},
                source="pattern_match",
                confidence=pattern_result["confidence"],
            )]

        return detect_by_llm(self, column_ref, values, context)

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
                confidence=0.85,
            ))

        return assertions

    def expand_synonyms(
        self,
        column_ref: str,
        terms: list[dict[str, str]],
        detected_vocab: str | None = None,
    ) -> list[Assertion]:
        """Generate synonym assertions for decoded terms via LLM."""
        synonyms = self._expand_via_llm(column_ref, terms)
        return build_synonym_assertions(
            self, column_ref, synonyms, detected_vocab,
        )

    def _expand_via_llm(
        self, subject_ref: str, terms: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        return expand_via_llm(self, subject_ref, terms)

    def process_column(
        self,
        column_ref: str,
        values: list[str],
        decoded_values: list[dict[str, str]] | None = None,
        context: VocabColumnContext | None = None,
    ) -> list[Assertion]:
        """Full L3 pipeline for a single column."""
        assertions: list[Assertion] = []

        assertions.extend(
            self.detect_vocabulary(column_ref, values, context)
        )
        apply_agreement_boost(assertions, context)

        detected_vocab = _extract_detected_vocab(assertions)
        if should_infer_hierarchy(context, detected_vocab):
            assertions.extend(
                self.infer_value_hierarchy(column_ref, values)
            )

        if decoded_values:
            assertions.extend(
                self.expand_synonyms(column_ref, decoded_values, detected_vocab)
            )

        return assertions


def _extract_detected_vocab(assertions: list[Assertion]) -> str | None:
    for a in assertions:
        if a.predicate == AssertionPredicate.VOCABULARY_MATCH:
            return a.payload.get("value")
    return None
