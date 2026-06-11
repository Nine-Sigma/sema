"""Term-hit expansion: vocabulary resolution, fan-out, ambiguity.

Term identity is {vocabulary_name, code}. When a hit's vocabulary is
unknown and its code exists in several vocabularies, one candidate is
emitted per vocabulary (capped), each expanded within its own
vocabulary, and every artifact carries its ambiguity group — retrieval
never silently picks among alternatives; downstream ranking (or a
council) selects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sema.graph.queries import CypherQueries
from sema.graph.term_identity_utils import UNSCOPED_VOCAB
from sema.log import logger

if TYPE_CHECKING:
    from sema.pipeline.retrieval import RetrievalEngine

MAX_VOCABULARY_FANOUT = 3


@dataclass(frozen=True)
class VocabResolution:
    """Vocabularies a term hit may belong to.

    ``failed`` means the lookup itself errored: the caller must emit a
    bare term and never fall back to unscoped expansion, which would
    silently match every vocabulary sharing the code.
    """
    names: tuple[str, ...]
    failed: bool = False


def _expand_ancestry(
    engine: RetrievalEngine,
    names_or_codes: list[str],
    vocabulary_name: str | None = None,
) -> list[dict[str, Any]]:
    """Expand Term ancestry via PARENT_OF traversal.

    Accepts term codes with optional vocabulary scope.
    """
    ancestry: list[dict[str, Any]] = []
    for code in names_or_codes:
        try:
            results = engine._run_query(
                CypherQueries.expand_ancestry(),
                code=code,
                vocabulary_name=vocabulary_name,
            )
            ancestry.extend(results)
        except Exception:
            pass
    return ancestry


def _expand_term_hit(
    engine: RetrievalEngine, hit: dict[str, Any]
) -> list[dict[str, Any]]:
    """Expand a Term vector hit using graph edges.

    Emits one candidate per resolved vocabulary; on resolver failure
    only a bare term candidate, with no expansion.
    """
    code = hit.get("code", hit.get("name", ""))
    label = hit.get("label", "")
    resolution = _term_vocabularies(engine, hit, code)
    if resolution.failed:
        return [_term_candidate(hit, code, label, None)]

    results: list[dict[str, Any]] = []
    for vocab in resolution.names:
        results.extend(
            _expand_term_for_vocab(engine, hit, code, label, vocab)
        )
    if len(resolution.names) > 1:
        _mark_ambiguous(results, code)
    return results


def _term_vocabularies(
    engine: RetrievalEngine, hit: dict[str, Any], code: str,
) -> VocabResolution:
    """Resolve a hit's candidate vocabularies, capped at the fan-out
    budget. Terms with no vocabulary live in the _unscoped namespace,
    so unresolved codes scope there instead of matching everywhere."""
    vocabulary_name = hit.get("vocabulary_name", "")
    if vocabulary_name:
        return VocabResolution(names=(vocabulary_name,))
    if not code:
        return VocabResolution(names=(UNSCOPED_VOCAB,))
    try:
        rows = engine._run_query(
            CypherQueries.find_vocabularies_for_term(), code=code,
        )
    except Exception as exc:
        logger.warning(
            "Vocabulary lookup failed for term code {!r}: {}",
            code, exc,
        )
        return VocabResolution(names=(), failed=True)

    names = tuple(
        r["vocabulary_name"] for r in rows if r.get("vocabulary_name")
    )
    if not names:
        return VocabResolution(names=(UNSCOPED_VOCAB,))
    if len(names) > MAX_VOCABULARY_FANOUT:
        logger.warning(
            "Term code {!r} exists in {} vocabularies; "
            "expanding only the first {}",
            code, len(names), MAX_VOCABULARY_FANOUT,
        )
        names = names[:MAX_VOCABULARY_FANOUT]
    return VocabResolution(names=names)


def _mark_ambiguous(
    artifacts: list[dict[str, Any]], group: str,
) -> None:
    for artifact in artifacts:
        artifact["ambiguous"] = True
        artifact["ambiguity_group"] = group


def _display_vocab(vocab: str | None) -> str | None:
    """Vocabulary as shown to consumers; the sentinel stays internal."""
    return vocab if vocab and vocab != UNSCOPED_VOCAB else None


def _term_candidate(
    hit: dict[str, Any], code: str, label: str, vocab: str | None,
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "type": "term",
        "code": code,
        "label": label,
        "score": hit.get("final_score", 0.0),
        "source": "retrieval",
        "status": hit.get("status", "auto"),
        "confidence": hit.get("confidence", 0.5),
        "confidence_policy": "semantic",
    }
    display = _display_vocab(vocab)
    if display:
        candidate["vocabulary"] = display
    return candidate


def _expand_term_for_vocab(
    engine: RetrievalEngine,
    hit: dict[str, Any],
    code: str,
    label: str,
    vocab: str | None,
) -> list[dict[str, Any]]:
    """One term candidate plus its vocabulary-scoped expansions."""
    hit_confidence = hit.get("confidence", 0.5)
    hit_status = hit.get("status", "auto")
    results: list[dict[str, Any]] = [
        _term_candidate(hit, code, label, vocab)
    ]
    if not code:
        return results

    results.extend(
        _expand_term_governed_values(
            engine, code, hit_status, hit_confidence,
            vocabulary_name=vocab,
        )
    )

    display = _display_vocab(vocab)
    for a in _expand_ancestry(engine, [code], vocabulary_name=vocab):
        results.append({
            "type": "ancestry",
            "code": a.get("code", ""),
            "label": a.get("label", ""),
            "vocabulary": a.get("vocabulary_name", display),
            "parent_code": a.get("parent_code", code),
            "source": "retrieval_ancestry",
            "status": hit_status,
            "confidence": hit_confidence,
            "confidence_policy": "semantic",
        })
    return results


def _expand_term_governed_values(
    engine: RetrievalEngine,
    code: str,
    status: str,
    confidence: float,
    vocabulary_name: str | None = None,
) -> list[dict[str, Any]]:
    """Traverse Term→MEMBER_OF→ValueSet←HAS_VALUE_SET←Column.

    Filters by vocabulary: codes are unique only within a vocabulary,
    so an unscoped match can cross vocabularies.
    """
    try:
        vs_results = engine._run_query(
            CypherQueries.find_value_sets_for_term(),
            code=code,
            vocabulary_name=vocabulary_name or None,
        )
    except Exception:
        return []

    display = _display_vocab(vocabulary_name)
    values: list[dict[str, Any]] = []
    for vs in vs_results:
        col = vs.get("column_name", "")
        table = vs.get("table_name", "")
        if not col or not table:
            continue
        entry: dict[str, Any] = {
            "type": "value",
            "property_name": col,
            "column": col,
            "table": table,
            "code": code,
            "label": code,
            "source": "retrieval_term_expansion",
            "status": status,
            "confidence": confidence,
            "confidence_policy": "semantic",
        }
        if display:
            entry["vocabulary"] = display
        values.append(entry)
    return values
