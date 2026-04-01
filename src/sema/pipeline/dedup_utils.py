"""Deduplication, ranking, and normalization helpers for retrieval.

Extracted from retrieval_utils.py to keep files under 400 lines.
"""
from __future__ import annotations

import json
import re
from types import MappingProxyType
from typing import Any

from sema.models.constants import MATCH_TYPE_BOOST

_STOP_WORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all",
    "can", "her", "was", "one", "our", "out", "has", "have",
    "what", "who", "how", "when", "where", "which", "with",
    "that", "this", "from", "they", "been", "will", "each",
    "make", "like", "most", "many", "some",
})

_TOKEN_SPLIT_RE = re.compile(r"[\s\-_/.,;:!?()]+")


def merge_and_rank_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank candidates using multi-signal scoring.

    Excludes DEPRECATED nodes from results.
    """
    active = [
        c for c in candidates
        if c.get("status", "ACTIVE") != "DEPRECATED"
    ]
    for c in active:
        base_score = c.get("score", 0.5)
        confidence = c.get("confidence", 0.5)
        match_boost = MATCH_TYPE_BOOST.get(
            c.get("match_type", "vector"), 0.0
        )
        c["final_score"] = (
            base_score * 0.4
            + confidence * 0.3
            + match_boost
            + 0.3
        )
    return sorted(
        active, key=lambda c: c["final_score"], reverse=True,
    )


def tokenize_query(query: str) -> list[str]:
    """Tokenize a query for lexical search.

    Lowercase, split on whitespace/punctuation, drop stop words
    and tokens shorter than 3 characters.
    """
    raw_tokens = _TOKEN_SPLIT_RE.split(query.lower().strip())
    return [
        t for t in raw_tokens
        if len(t) >= 3 and t not in _STOP_WORDS
    ]


def normalize_vector_hit(
    hit: dict[str, Any], node_type: str,
) -> dict[str, Any]:
    """Normalize a raw vector search hit into a SeedHit shape."""
    return {
        **hit,
        "node_type": node_type,
        "match_type": hit.get("match_type", "vector"),
        "score": hit.get("score", 0.0),
        "confidence": hit.get("confidence", 0.5),
        "status": hit.get("status", "auto"),
    }


def dedup_seeds(
    seeds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate seed hits by graph-node identity.

    Uses scoped identity when present, falls back to
    (node_type, normalized_name) for Properties and others.
    Keeps the highest-ranked hit per identity.
    """
    seen: dict[tuple[str, ...], dict[str, Any]] = {}
    for seed in seeds:
        key = _seed_identity_key(seed)
        existing = seen.get(key)
        if existing is None or seed.get(
            "final_score", 0.0
        ) > existing.get("final_score", 0.0):
            seen[key] = seed
    return list(seen.values())


def _seed_identity_key(
    seed: dict[str, Any],
) -> tuple[str, ...]:
    """Build a hashable identity key for a seed hit."""
    node_type = seed.get("node_type", "")

    ds_id = seed.get("datasource_id", "")
    if node_type == "property" and ds_id:
        ck = seed.get("column_key", "")
        if ck:
            return (node_type, ds_id, ck)

    if node_type == "term":
        vocab = seed.get("vocabulary_name", "")
        code = seed.get("code", seed.get("name", ""))
        if vocab and code:
            return (node_type, vocab, code)
        return (node_type, code.lower())

    if node_type == "alias":
        tk = seed.get("target_key", "")
        text = seed.get("text", seed.get("name", ""))
        if tk:
            return (node_type, tk, text.lower())
        return (node_type, text.lower())

    name = seed.get("name", "").lower()
    return (node_type, name)


def dedup_artifacts(
    artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate expanded artifacts by type-specific keys.

    Keeps the highest-confidence artifact per identity.
    """
    seen: dict[tuple[str, ...], dict[str, Any]] = {}
    for art in artifacts:
        key = _artifact_identity_key(art)
        existing = seen.get(key)
        if existing is None or art.get(
            "confidence", 0.0
        ) > existing.get("confidence", 0.0):
            seen[key] = art
    return list(seen.values())


def _artifact_identity_key(
    art: dict[str, Any],
) -> tuple[str, ...]:
    """Build a hashable identity for an expanded artifact."""
    art_type = art.get("type", "")

    if art_type == "entity":
        return (
            art_type,
            art.get("catalog", ""),
            art.get("schema", ""),
            art.get("table", ""),
        )
    if art_type == "property":
        return (
            art_type,
            art.get("entity_name", ""),
            art.get("name", ""),
            art.get("physical_column", art.get("column", "")),
        )
    if art_type == "value":
        return (
            art_type,
            art.get("table", ""),
            art.get("column", ""),
            art.get("code", ""),
        )
    if art_type == "join":
        return (
            art_type,
            art.get("from_table", ""),
            art.get("to_table", ""),
            _serialize_join_shape(art),
        )
    if art_type == "metric":
        return (art_type, art.get("name", ""))
    if art_type == "ancestry":
        return (
            art_type,
            art.get("code", ""),
            art.get("parent_code", ""),
        )
    return (art_type, art.get("text", art.get("name", "")))


def _serialize_join_shape(art: dict[str, Any]) -> str:
    """Deterministic hashable string for join identity."""
    snippet = art.get("sql_snippet")
    if snippet:
        return str(snippet)
    preds = art.get("join_predicates")
    if preds is None:
        return ""
    if isinstance(preds, str):
        return preds
    return json.dumps(preds, sort_keys=True, default=str)
