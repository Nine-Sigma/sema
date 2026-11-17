"""Helper functions for the retrieval pipeline.

Extracted from retrieval.py to keep the module focused on the
RetrievalEngine class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neo4j.exceptions import DriverError, Neo4jError

from sema.graph.queries import CypherQueries
from sema.log import logger
from sema.pipeline.dedup_utils import (
    dedup_artifacts as _dedup_artifacts,
    dedup_seeds as _dedup_seeds,
    merge_and_rank_candidates,
    _seed_identity_key as seed_identity_key,
    normalize_vector_hit as _normalize_vector_hit,
    tokenize_query,
)
from sema.pipeline.term_expansion_utils import (
    _expand_ancestry,
    _expand_term_governed_values,
    _expand_term_hit,
)

if TYPE_CHECKING:
    from sema.pipeline.retrieval import RetrievalEngine

_PROPERTY_OWNER_CONFIDENCE = 0.6

__all__ = [
    "_dedup_artifacts",
    "_dedup_seeds",
    "_expand_ancestry",
    "_expand_term_governed_values",
    "_expand_term_hit",
    "_normalize_vector_hit",
    "merge_and_rank_candidates",
    "tokenize_query",
]


def _expand_physical(
    engine: RetrievalEngine, entity_names: list[str]
) -> list[dict[str, Any]]:
    scope = getattr(engine, "_scope", None)
    physical: list[dict[str, Any]] = []
    for name in entity_names:
        try:
            physical.extend(_resolve_physical_for(engine, name, scope))
        except Exception:
            pass
    return physical


def _resolve_physical_for(
    engine: RetrievalEngine, name: str, scope: Any,
) -> list[dict[str, Any]]:
    if scope is not None and scope.is_active():
        return engine._run_query(
            CypherQueries.resolve_physical_mapping_scoped(),
            entity_name=name,
            model_role=scope.model_role,
            schema_name=scope.schema_name,
        )
    return engine._run_query(
        CypherQueries.resolve_physical_mapping(), entity_name=name,
    )


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
    except (Neo4jError, DriverError):
        logger.error(f"Join path read failed for tables {table_names}")
        raise


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
                    schema_name=r.get("schema_name"),
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


def _expand_property_hit(
    engine: RetrievalEngine, hit: dict[str, Any]
) -> list[dict[str, Any]]:
    """Expand a Property vector hit using graph edges.

    Traverses: owning Entity (HAS_PROPERTY), physical Column
    (PROPERTY_ON_COLUMN), governed ValueSet (HAS_VALUE_SET),
    Vocabulary (CLASSIFIED_AS).
    """
    name = hit.get("name", "")
    prop_candidate: dict[str, Any] = {
        "type": "property",
        "name": name,
        "entity_name": hit.get("entity_name", ""),
        "description": hit.get("description"),
        "score": hit.get("final_score", 0.0),
        "source": "retrieval",
        "status": hit.get("status", "auto"),
        "confidence": hit.get("confidence", 0.5),
        "confidence_policy": "semantic",
        **_property_vocabulary(engine, hit),
    }
    owners = _property_owners(engine, hit)
    prop_candidate.update(_physical_column(owners, name))
    return [prop_candidate, *owners]


def _property_vocabulary(
    engine: RetrievalEngine, hit: dict[str, Any],
) -> dict[str, Any]:
    """Vocabulary classification via CLASSIFIED_AS, when the hit is scoped."""
    datasource_id = hit.get("datasource_id", "")
    column_key = hit.get("column_key", "")
    if not (datasource_id and column_key):
        return {}
    try:
        vocab_results = engine._run_query(
            CypherQueries.find_property_vocabulary(),
            ds=datasource_id, ck=column_key,
        )
    except Exception:
        return {}
    if not vocab_results:
        return {}
    return {"vocabulary": vocab_results[0].get("vocabulary_name")}


def _property_owners(
    engine: RetrievalEngine, hit: dict[str, Any],
) -> list[dict[str, Any]]:
    """The owning entity's tables. The property the vector index ranks nearest
    the question brings its table at the property's own confidence; any other
    property hit's table keeps the expansion confidence, below the semantic
    threshold, so it does not enter alone."""
    entity_name = hit.get("entity_name", "")
    if not entity_name:
        return []
    confidence = (
        hit.get("confidence", 0.5)
        if hit.get("nearest_property")
        else _PROPERTY_OWNER_CONFIDENCE
    )
    return [
        {
            "type": "entity",
            "name": entity_name,
            "table": r.get("table_name", ""),
            "schema": r.get("schema_name", ""),
            "catalog": r.get("catalog", ""),
            "description": None,
            "confidence": confidence,
            "source": "retrieval_property_expansion",
            "columns": r.get("columns", []),
            "status": hit.get("status", "auto"),
            "confidence_policy": "semantic",
        }
        for r in _expand_physical(engine, [entity_name])
    ]


def mark_nearest_property(
    vector_hits: list[dict[str, Any]], seeds: list[dict[str, Any]],
) -> None:
    """Flag the seed of the property hit the vector index ranks nearest the
    question. An index returns its nearest nodes for any question, and only
    their order, never the level of their similarity, holds across embedders."""
    properties = [h for h in vector_hits if h.get("node_type") == "property"]
    top = max((h.get("score", 0.0) for h in properties), default=None)
    nearest = {seed_identity_key(h) for h in properties if h.get("score", 0.0) == top}
    for seed in seeds:
        if seed.get("node_type") == "property" and seed_identity_key(seed) in nearest:
            seed["nearest_property"] = True


def _physical_column(
    owners: list[dict[str, Any]], name: str,
) -> dict[str, str]:
    """Where the property sits: the first owning table with a column for it."""
    for owner in owners:
        for col in owner.get("columns", []):
            if col.get("property") == name and col.get("column"):
                return {
                    "physical_table": owner["table"],
                    "physical_column": col["column"],
                }
    return {}


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
