"""Helper functions for the retrieval pipeline.

Extracted from retrieval.py to keep the module focused on the
RetrievalEngine class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sema.graph.queries import CypherQueries
from sema.models.constants import MATCH_TYPE_BOOST

if TYPE_CHECKING:
    from sema.pipeline.retrieval import RetrievalEngine


def merge_and_rank_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank candidates using multi-signal scoring."""
    for c in candidates:
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
    return sorted(candidates, key=lambda c: c["final_score"], reverse=True)


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


def _expand_transformations(
    engine: RetrievalEngine, table_names: list[str]
) -> list[dict[str, Any]]:
    if not table_names:
        return []
    try:
        return engine._run_query(
            CypherQueries.expand_transformations(),
            table_names=table_names,
        )
    except Exception:
        return []


def _expand_values(
    engine: RetrievalEngine, physical: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for r in physical:
        for col in r.get("columns", []):
            if col.get("semantic_type") == "categorical":
                try:
                    vs_name = (
                        f"{r['table_name']}_{col['column']}_values"
                    )
                    vs_results = engine._run_query(
                        CypherQueries.expand_value_set(),
                        value_set_name=vs_name,
                    )
                    values.extend([
                        {**v, "property": col["property"],
                         "column": col["column"],
                         "table": r["table_name"]}
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
            m = engine._run_query(
                CypherQueries.expand_metrics(),
                entity_name=name,
            )
            metrics.extend(m)
        except Exception:
            pass
    return metrics
