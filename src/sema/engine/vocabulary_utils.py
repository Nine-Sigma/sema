from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
)

if TYPE_CHECKING:
    from sema.engine.vocabulary import VocabColumnContext, VocabularyEngine

logger = logging.getLogger(__name__)


def _build_context_block(context: VocabColumnContext | None) -> str:
    """Build a context section for LLM prompts from L2-derived context."""
    if not context:
        return ""
    lines: list[str] = []
    if context.table_name:
        lines.append(f"Table: {context.table_name}")
    if context.entity_name:
        lines.append(f"Entity: {context.entity_name}")
    if context.column_name:
        lines.append(f"Column: {context.column_name}")
    if context.property_name:
        lines.append(f"Property: {context.property_name}")
    if context.semantic_type:
        lines.append(f"Semantic type: {context.semantic_type}")
    if context.vocabulary_guess:
        lines.append(
            f"L2 suggestion: {context.vocabulary_guess} "
            f"(confidence {context.vocabulary_guess_confidence:.2f})"
        )
    if not lines:
        return ""
    return "Context:\n" + "\n".join(lines) + "\n\n"


def _build_vocab_prompt(
    values: list[str], context: VocabColumnContext | None = None,
) -> str:
    """Build the vocabulary detection prompt with optional L2 context."""
    context_block = _build_context_block(context)
    return (
        f"{context_block}"
        f"Given these values from a database column: "
        f"{values[:20]}\n"
        f"What standard vocabulary or coding system do they "
        f"belong to?\n"
        f'Return JSON: {{"vocabulary": "name", '
        f'"confidence": 0.0-1.0}}\n'
        f"Return ONLY valid JSON."
    )


def detect_by_llm(
    engine: VocabularyEngine,
    subject_ref: str,
    values: list[str],
    context: VocabColumnContext | None = None,
) -> list[Assertion]:
    if engine._llm_client and values:
        return detect_by_llm_client(engine, subject_ref, values, context)
    elif engine._llm and values:
        return detect_by_llm_legacy(engine, subject_ref, values, context)
    return []


def detect_by_llm_client(
    engine: VocabularyEngine,
    subject_ref: str,
    values: list[str],
    context: VocabColumnContext | None = None,
) -> list[Assertion]:
    from sema.llm_client import VocabularyDetection

    prompt = _build_vocab_prompt(values, context)
    result = engine._llm_client.invoke(
        prompt,
        VocabularyDetection,
        table_ref=subject_ref,
        stage_name="L3 vocabulary",
    )
    if result.vocabulary:
        confidence = compute_vocab_confidence(context, result.vocabulary, values)
        return [engine._make_assertion(
            subject_ref,
            AssertionPredicate.VOCABULARY_MATCH,
            {"value": result.vocabulary},
            source="llm_interpretation",
            confidence=confidence,
        )]
    return []


def detect_by_llm_legacy(
    engine: VocabularyEngine,
    subject_ref: str,
    values: list[str],
    context: VocabColumnContext | None = None,
) -> list[Assertion]:
    try:
        prompt = _build_vocab_prompt(values, context)
        response = engine._llm.invoke(prompt)
        content = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )
        content = _strip_markdown_fences(content)
        result = json.loads(content)
        if result.get("vocabulary"):
            confidence = compute_vocab_confidence(
                context, result["vocabulary"], values,
            )
            return [engine._make_assertion(
                subject_ref,
                AssertionPredicate.VOCABULARY_MATCH,
                {"value": result["vocabulary"]},
                source="llm_interpretation",
                confidence=confidence,
            )]
    except Exception as e:
        logger.warning(
            "LLM vocabulary detection failed for %s: %s",
            subject_ref, e,
        )
    return []


def _strip_markdown_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [
            line for line in lines
            if not line.strip().startswith("```")
        ]
        content = "\n".join(lines).strip()
    return content


def _normalize_vocab_name(name: str) -> str:
    """Normalize vocabulary names for fuzzy comparison."""
    s = name.lower().strip()
    s = re.sub(r"\s*\d+(st|nd|rd|th)\s+edition\s*", "", s)
    for suffix in ("staging", "classification", "codes", "system"):
        s = s.replace(suffix, "").strip()
    return s


def vocabs_agree(a: str, b: str) -> bool:
    return _normalize_vocab_name(a) == _normalize_vocab_name(b)


def apply_agreement_boost(
    assertions: list[Assertion],
    context: VocabColumnContext | None,
) -> None:
    """Boost confidence when L2 and L3 agree on vocabulary (in-place)."""
    if not context or not context.vocabulary_guess:
        return
    for assertion in assertions:
        if assertion.predicate != AssertionPredicate.VOCABULARY_MATCH:
            continue
        l3_vocab = assertion.payload.get("value", "")
        if vocabs_agree(l3_vocab, context.vocabulary_guess):
            assertion.confidence = min(assertion.confidence + 0.1, 1.0)


NON_HIERARCHICAL_TYPES = frozenset({
    "numeric", "temporal", "identifier", "free_text",
})

HIERARCHICAL_VOCABULARIES = frozenset({
    "icd-10", "atc", "cpt", "snomed", "ajcc",
    "tnm", "oncotree", "loinc", "meddra",
})


def should_infer_hierarchy(
    context: VocabColumnContext | None,
    detected_vocab: str | None,
) -> bool:
    """Three-gate filter: skip non-categorical, run for known hierarchical vocabs."""
    if not context:
        return False
    if context.semantic_type and context.semantic_type.lower() in NON_HIERARCHICAL_TYPES:
        return False
    if detected_vocab:
        return _normalize_vocab_name(detected_vocab) in HIERARCHICAL_VOCABULARIES
    return False


_VOCAB_COLUMN_KEYWORDS = frozenset({
    "code", "cd", "type", "stage", "class", "category", "dx", "icd", "cpt",
})


def compute_vocab_confidence(
    context: VocabColumnContext | None,
    detected_vocab: str,
    values: list[str],
) -> float:
    """Compute calibrated confidence from multiple signals."""
    base = 0.5
    if context and context.vocabulary_guess:
        if vocabs_agree(context.vocabulary_guess, detected_vocab):
            base += 0.15
    if context and context.column_name:
        if any(kw in context.column_name.lower() for kw in _VOCAB_COLUMN_KEYWORDS):
            base += 0.1
    if len(values) >= 5:
        base += 0.05
    return min(base, 1.0)


def _alias_expand_prompt(labels: list[str]) -> str:
    """Build LLM prompt for term alias expansion."""
    return (
        f"For each of these terms, provide common aliases "
        f"and abbreviations:\n"
        f"{json.dumps(labels)}\n"
        f'Return JSON: {{"synonyms": [{{"term": "X", '
        f'"synonyms": ["y", "z"]}}]}}\n'
        f"Return ONLY valid JSON."
    )


def expand_via_llm(
    engine: VocabularyEngine, subject_ref: str, terms: list[dict[str, str]]
) -> list[dict[str, Any]]:
    if engine._llm_client and terms:
        from sema.llm_client import SynonymExpansion

        labels = [t["label"] for t in terms[:20]]
        prompt = _alias_expand_prompt(labels)
        result = engine._llm_client.invoke(
            prompt,
            SynonymExpansion,
            table_ref=subject_ref,
            stage_name="L3 vocabulary",
        )
        return [
            {"term": item.get("term", ""), "synonyms": item.get("synonyms", [])}
            for item in result.synonyms
        ]
    elif engine._llm and terms:
        try:
            labels = [t["label"] for t in terms[:20]]
            prompt = _alias_expand_prompt(labels)
            response = engine._llm.invoke(prompt)
            content = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
            result = json.loads(content)
            return [
                {"term": item.get("term", ""), "synonyms": item.get("synonyms", [])}
                for item in result.get("synonyms", [])
            ]
        except Exception as e:
            logger.warning(
                "LLM alias expansion failed for %s: %s", subject_ref, e
            )
    return []


def build_synonym_assertions(
    engine: VocabularyEngine,
    subject_ref: str,
    synonyms: list[dict[str, Any]],
    detected_vocab: str | None = None,
) -> list[Assertion]:
    """Emit HAS_ALIAS assertions for term synonym expansions.

    The first alias for each term gets is_preferred=True; subsequent ones False.
    """
    confidence = 0.75 if detected_vocab else 0.65
    assertions: list[Assertion] = []
    for item in synonyms:
        term_name = item.get("term", "")
        for i, syn in enumerate(item.get("synonyms", [])):
            assertions.append(engine._make_assertion(
                subject_ref,
                AssertionPredicate.HAS_ALIAS,
                {"term": term_name, "value": syn, "is_preferred": i == 0},
                source="llm_interpretation",
                confidence=confidence,
            ))
    return assertions
