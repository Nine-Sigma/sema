from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
)

if TYPE_CHECKING:
    from sema.engine.vocabulary import VocabularyEngine

logger = logging.getLogger(__name__)


def detect_by_llm(
    engine: VocabularyEngine, subject_ref: str, values: list[str]
) -> list[Assertion]:
    if engine._llm_client and values:
        return detect_by_llm_client(engine, subject_ref, values)
    elif engine._llm and values:
        return detect_by_llm_legacy(engine, subject_ref, values)
    return []


def detect_by_llm_client(
    engine: VocabularyEngine, subject_ref: str, values: list[str]
) -> list[Assertion]:
    from sema.llm_client import (
        VocabularyDetection,
    )
    assertions: list[Assertion] = []
    prompt = (
        f"Given these values from a database column: "
        f"{values[:20]}\n"
        f"What standard vocabulary or coding system do they "
        f"belong to?\n"
        f'Return JSON: {{"vocabulary": "name", '
        f'"confidence": 0.0-1.0}}\n'
        f"Return ONLY valid JSON."
    )
    result = engine._llm_client.invoke(
        prompt,
        VocabularyDetection,
        table_ref=subject_ref,
        stage_name="L3 vocabulary",
    )
    if result.vocabulary:
        assertions.append(engine._make_assertion(
            subject_ref,
            AssertionPredicate.VOCABULARY_MATCH,
            {"value": result.vocabulary},
            source="llm_interpretation",
            confidence=result.confidence,
        ))
    return assertions


def detect_by_llm_legacy(
    engine: VocabularyEngine, subject_ref: str, values: list[str]
) -> list[Assertion]:
    assertions: list[Assertion] = []
    try:
        prompt = (
            f"Given these values from a database column: "
            f"{values[:20]}\n"
            f"What standard vocabulary or coding system do "
            f"they belong to?\n"
            f'Return JSON: {{"vocabulary": "name", '
            f'"confidence": 0.0-1.0}}\n'
            f"Return ONLY valid JSON."
        )
        response = engine._llm.invoke(prompt)
        content = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [
                line
                for line in lines
                if not line.strip().startswith("```")
            ]
            content = "\n".join(lines).strip()
        result = json.loads(content)
        if result.get("vocabulary"):
            assertions.append(engine._make_assertion(
                subject_ref,
                AssertionPredicate.VOCABULARY_MATCH,
                {"value": result["vocabulary"]},
                source="llm_interpretation",
                confidence=result.get("confidence", 0.6),
            ))
    except Exception as e:
        logger.warning(
            f"LLM vocabulary detection failed for "
            f"{subject_ref}: {e}"
        )
    return assertions


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
        from sema.llm_client import (
            SynonymExpansion,
        )
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
    engine: VocabularyEngine, subject_ref: str, synonyms: list[dict[str, Any]]
) -> list[Assertion]:
    """Emit HAS_ALIAS assertions for term synonym expansions.

    The first alias for each term gets is_preferred=True; subsequent ones False.
    """
    assertions: list[Assertion] = []
    for item in synonyms:
        term_name = item.get("term", "")
        for i, syn in enumerate(item.get("synonyms", [])):
            assertions.append(engine._make_assertion(
                subject_ref,
                AssertionPredicate.HAS_ALIAS,
                {"term": term_name, "value": syn, "is_preferred": i == 0},
                source="llm_interpretation",
                confidence=0.7,
            ))
    return assertions
