from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sema.engine.vocabulary import VocabColumnContext


@dataclass(frozen=True)
class VocabPattern:
    """Declarative vocabulary pattern for regex-based detection.

    Patterns without context_keywords match unconditionally.
    Patterns with context_keywords only match when column context
    contains at least one keyword (checked against column_name,
    table_name, entity_name, property_name, semantic_type,
    and vocabulary_guess).

    domain_affinity gates matching by warehouse domain when a
    WarehouseProfile is available. When no profile is provided,
    domain_affinity is ignored (falls back to context_keywords only).
    """

    name: str
    pattern: re.Pattern[str]
    context_keywords: frozenset[str] = field(default_factory=frozenset)
    domain_affinity: frozenset[str] = field(default_factory=frozenset)

    def matches(
        self,
        values: list[str],
        context: VocabColumnContext | None = None,
        threshold: float = 0.6,
        warehouse_domains: dict[str, float] | None = None,
    ) -> float | None:
        """Return match ratio if values match this pattern, else None."""
        # Domain affinity gating (skip when no profile available)
        if self.domain_affinity and warehouse_domains:
            if not any(
                warehouse_domains.get(d, 0) > 0.1
                for d in self.domain_affinity
            ):
                return None

        if self.context_keywords:
            if not _has_context_keyword(context, self.context_keywords):
                return None
        ratio = _match_ratio(values, self.pattern)
        return ratio if ratio >= threshold else None


def _match_ratio(values: list[str], pattern: re.Pattern[str]) -> float:
    match_count = sum(1 for v in values if pattern.match(str(v).strip()))
    return match_count / len(values) if values else 0


def _has_context_keyword(
    context: VocabColumnContext | None,
    keywords: frozenset[str],
) -> bool:
    if not context:
        return False
    text = _context_text(context)
    return any(kw in text for kw in keywords)


def _context_text(context: VocabColumnContext) -> str:
    parts = (
        context.column_name,
        context.table_name,
        context.entity_name,
        context.property_name,
        context.semantic_type,
        context.vocabulary_guess,
    )
    return " ".join(part.lower() for part in parts if part)


def detect_first_match(
    patterns: list[VocabPattern],
    values: list[str],
    context: VocabColumnContext | None = None,
    warehouse_domains: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Return the first matching vocabulary pattern result, or None."""
    if not values:
        return None
    for vp in patterns:
        ratio = vp.matches(values, context, warehouse_domains=warehouse_domains)
        if ratio is not None:
            return {
                "vocabulary": vp.name,
                "confidence": min(0.9 + (ratio - 0.6) * 0.25, 1.0),
                "match_ratio": ratio,
            }
    return None
