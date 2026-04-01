"""Helper functions for the retrieval pipeline.

Extracted from retrieval.py to keep the module focused on the
RetrievalEngine class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sema.graph.queries import CypherQueries
from sema.log import logger
from sema.pipeline.dedup_utils import (
    dedup_artifacts as _dedup_artifacts,
    dedup_seeds as _dedup_seeds,
    merge_and_rank_candidates,
    normalize_vector_hit as _normalize_vector_hit,
    tokenize_query,
)

if TYPE_CHECKING:
    from sema.pipeline.retrieval import RetrievalEngine

__all__ = [
    "_dedup_artifacts",
    "_dedup_seeds",
    "_normalize_vector_hit",
    "merge_and_rank_candidates",
    "tokenize_query",
]


def _expand_physical(
    engine: RetrievalEngine, entity_names: list[str]
) -> list[dict[str, Any]]:
    physical: list[dict[str, Any]] = []
    for name in entity_names:
        try:
            results = engine._run_query(
                CypherQueries.resolve_physical_mapping(),
                entity_name=name,
            )
            physical.extend(results)
        except Exception:
            pass
    return physical


def _expand_joins(
    engine: RetrievalEngine, table_names: list[str]
) -> list[dict[str, Any]]:
    if not table_names:
        return []
    try:
        return engine._run_query(
            CypherQueries.find_join_paths(),
            table_names=table_names,
        )
    except Exception:
        return []


def _expand_values(
    engine: RetrievalEngine, physical: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Expand governed values via HAS_VALUE_SET graph edges."""
    values: list[dict[str, Any]] = []
    for r in physical:
        for col in r.get("columns", []):
            if col.get("semantic_type") != "categorical":
                continue
            col_name = col.get("column", "")
            table_name = r.get("table_name", "")
            if not col_name or not table_name:
                continue
            try:
                vs_results = engine._run_query(
                    CypherQueries.find_value_set_members_by_column(),
                    column_name=col_name,
                    table_name=table_name,
                )
                values.extend([
                    {**v, "property": col["property"],
                     "column": col_name,
                     "table": table_name}
                    for v in vs_results
                ])
            except Exception:
                pass
    return values


def _expand_metrics(
    engine: RetrievalEngine, entity_names: list[str]
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for name in entity_names:
        try:
            results = engine._run_query(
                CypherQueries.expand_metrics(),
                entity_name=name,
            )
            for m in results:
                metrics.append({
                    "name": m.get("name", ""),
                    "description": m.get("description"),
                    "formula": m.get("formula"),
                    "confidence": m.get("confidence", 0.5),
                    "aggregates": m.get("aggregates", []),
                    "filters": m.get("filters", []),
                    "grains": m.get("grains", []),
                    "source": "retrieval",
                    "status": "auto",
                    "confidence_policy": "semantic",
                })
        except Exception:
            pass
    return metrics


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


def _expand_property_hit(
    engine: RetrievalEngine, hit: dict[str, Any]
) -> list[dict[str, Any]]:
    """Expand a Property vector hit using graph edges.

    Traverses: owning Entity (HAS_PROPERTY), physical Column
    (PROPERTY_ON_COLUMN), governed ValueSet (HAS_VALUE_SET),
    Vocabulary (CLASSIFIED_AS).
    """
    results: list[dict[str, Any]] = []
    name = hit.get("name", "")
    entity_name = hit.get("entity_name", "")
    datasource_id = hit.get("datasource_id", "")
    column_key = hit.get("column_key", "")

    hit_confidence = hit.get("confidence", 0.5)
    hit_status = hit.get("status", "auto")

    prop_candidate: dict[str, Any] = {
        "type": "property",
        "name": name,
        "entity_name": entity_name,
        "score": hit.get("final_score", 0.0),
        "source": "retrieval",
        "status": hit_status,
        "confidence": hit_confidence,
        "confidence_policy": "semantic",
    }

    # Try to get vocabulary classification via CLASSIFIED_AS
    if datasource_id and column_key:
        try:
            vocab_results = engine._run_query(
                CypherQueries.find_property_vocabulary(),
                ds=datasource_id, ck=column_key,
            )
            if vocab_results:
                prop_candidate["vocabulary"] = vocab_results[0].get(
                    "vocabulary_name"
                )
        except Exception:
            pass

    results.append(prop_candidate)

    # Expand owning entity if known
    if entity_name:
        physical = _expand_physical(engine, [entity_name])
        for r in physical:
            results.append({
                "type": "entity",
                "name": entity_name,
                "table": r.get("table_name", ""),
                "schema": r.get("schema_name", ""),
                "catalog": r.get("catalog", ""),
                "description": None,
                "confidence": 0.6,
                "source": "retrieval_property_expansion",
                "columns": r.get("columns", []),
                "status": hit_status,
                "confidence_policy": "semantic",
            })

    return results


def _expand_term_hit(
    engine: RetrievalEngine, hit: dict[str, Any]
) -> list[dict[str, Any]]:
    """Expand a Term vector hit using graph edges.

    Traverses: Vocabulary (IN_VOCABULARY), ValueSets (MEMBER_OF),
    associated Columns, ancestry (PARENT_OF*1..3).
    """
    results: list[dict[str, Any]] = []
    code = hit.get("code", hit.get("name", ""))
    label = hit.get("label", "")
    vocabulary_name = hit.get("vocabulary_name", "")
    hit_confidence = hit.get("confidence", 0.5)
    hit_status = hit.get("status", "auto")

    term_candidate: dict[str, Any] = {
        "type": "term",
        "code": code,
        "label": label,
        "score": hit.get("final_score", 0.0),
        "source": "retrieval",
        "status": hit_status,
        "confidence": hit_confidence,
        "confidence_policy": "semantic",
    }

    # Try to get vocabulary via IN_VOCABULARY
    if vocabulary_name:
        term_candidate["vocabulary"] = vocabulary_name
    elif code:
        try:
            vocab_results = engine._run_query(
                CypherQueries.find_vocabulary_for_term(),
                code=code,
            )
            if vocab_results:
                term_candidate["vocabulary"] = vocab_results[0].get(
                    "vocabulary_name"
                )
        except Exception:
            pass

    results.append(term_candidate)

    # Term→MEMBER_OF→ValueSet←HAS_VALUE_SET←Column
    if code:
        results.extend(
            _expand_term_governed_values(
                engine, code, hit_status, hit_confidence,
            )
        )

    # Expand ancestry from term code (vocabulary-scoped)
    vocab = term_candidate.get("vocabulary")
    if code:
        ancestry = _expand_ancestry(
            engine, [code], vocabulary_name=vocab,
        )
        for a in ancestry:
            results.append({
                "type": "ancestry",
                "code": a.get("code", ""),
                "label": a.get("label", ""),
                "vocabulary": a.get("vocabulary_name", vocab),
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
) -> list[dict[str, Any]]:
    """Traverse Term→MEMBER_OF→ValueSet←HAS_VALUE_SET←Column."""
    try:
        vs_results = engine._run_query(
            CypherQueries.find_value_sets_for_term(),
            code=code,
        )
    except Exception:
        return []

    values: list[dict[str, Any]] = []
    for vs in vs_results:
        col = vs.get("column_name", "")
        table = vs.get("table_name", "")
        if not col or not table:
            continue
        values.append({
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
        })
    return values


def _expand_alias_hit(
    engine: RetrievalEngine, hit: dict[str, Any]
) -> list[dict[str, Any]]:
    """Expand an Alias vector hit by dereferencing REFERS_TO target.

    Dispatches by target label (Entity/Property/Term).
    """
    results: list[dict[str, Any]] = []
    text = hit.get("text", hit.get("name", ""))
    target_name = hit.get("parent_name", hit.get("target_name", ""))
    hit_confidence = hit.get("confidence", 0.5)
    hit_status = hit.get("status", "auto")

    results.append({
        "type": "alias",
        "text": text,
        "target": target_name,
        "score": hit.get("final_score", 0.0),
        "source": "retrieval",
        "status": hit_status,
        "confidence": hit_confidence,
        "confidence_policy": "semantic",
    })

    if not text:
        return results

    # Dereference alias to get target labels
    target_labels: list[str] = []
    resolved_name = target_name
    try:
        deref = engine._run_query(
            CypherQueries.dereference_alias(), text=text,
        )
        if deref:
            target_labels = deref[0].get("target_labels", [])
            resolved_name = (
                deref[0].get("target_name") or target_name
            )
    except Exception:
        pass

    if not resolved_name:
        return results

    # Dispatch by target type
    if "Entity" in target_labels:
        for r in _expand_physical(engine, [resolved_name]):
            results.append({
                "type": "entity",
                "name": resolved_name,
                "table": r.get("table_name", ""),
                "schema": r.get("schema_name", ""),
                "catalog": r.get("catalog", ""),
                "description": None,
                "confidence": hit_confidence,
                "source": "retrieval_alias_expansion",
                "columns": r.get("columns", []),
                "status": hit_status,
                "confidence_policy": "semantic",
            })
    elif "Property" in target_labels:
        prop_hit = {
            "name": resolved_name,
            "entity_name": "",
            "final_score": hit.get("final_score", 0.0),
            "status": hit_status,
            "confidence": hit_confidence,
        }
        results.extend(_expand_property_hit(engine, prop_hit))
    elif "Term" in target_labels:
        term_hit = {
            "code": resolved_name,
            "label": resolved_name,
            "final_score": hit.get("final_score", 0.0),
            "status": hit_status,
            "confidence": hit_confidence,
        }
        results.extend(_expand_term_hit(engine, term_hit))
    elif target_labels:
        logger.debug(
            "Alias '{}' refers to unknown type: {}",
            text, target_labels,
        )
    else:
        # Fallback: try entity expansion (legacy behavior)
        for r in _expand_physical(engine, [resolved_name]):
            results.append({
                "type": "entity",
                "name": resolved_name,
                "table": r.get("table_name", ""),
                "schema": r.get("schema_name", ""),
                "catalog": r.get("catalog", ""),
                "description": None,
                "confidence": hit_confidence,
                "source": "retrieval_alias_expansion",
                "columns": r.get("columns", []),
                "status": hit_status,
                "confidence_policy": "semantic",
            })

    return results
