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
    return sorted(active, key=lambda c: c["final_score"], reverse=True)


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
    (PROPERTY_ON_COLUMN), governed ValueSet (STORED_IN),
    Vocabulary (CLASSIFIED_AS).
    """
    results: list[dict[str, Any]] = []
    name = hit.get("name", "")
    entity_name = hit.get("entity_name", "")
    datasource_id = hit.get("datasource_id", "")
    column_key = hit.get("column_key", "")

    prop_candidate: dict[str, Any] = {
        "type": "property",
        "name": name,
        "entity_name": entity_name,
        "score": hit.get("final_score", 0.0),
        "source": "retrieval",
    }

    # Try to get vocabulary classification via CLASSIFIED_AS
    if datasource_id and column_key:
        try:
            vocab_results = engine._run_query(
                "MATCH (p:Property {datasource_id: $ds, column_key: $ck})"
                "-[:CLASSIFIED_AS]->(v:Vocabulary) "
                "RETURN v.name AS vocabulary_name",
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

    term_candidate: dict[str, Any] = {
        "type": "term",
        "code": code,
        "label": label,
        "score": hit.get("final_score", 0.0),
        "source": "retrieval",
    }

    # Try to get vocabulary via IN_VOCABULARY
    if vocabulary_name:
        term_candidate["vocabulary"] = vocabulary_name
    elif code:
        try:
            vocab_results = engine._run_query(
                "MATCH (t:Term {code: $code})"
                "-[:IN_VOCABULARY]->(v:Vocabulary) "
                "RETURN v.name AS vocabulary_name LIMIT 1",
                code=code,
            )
            if vocab_results:
                term_candidate["vocabulary"] = vocab_results[0].get(
                    "vocabulary_name"
                )
        except Exception:
            pass

    results.append(term_candidate)

    # Expand ancestry from term code (vocabulary-scoped)
    vocab = term_candidate.get("vocabulary")
    if code:
        ancestry = _expand_ancestry(engine, [code], vocabulary_name=vocab)
        for a in ancestry:
            results.append({
                "type": "ancestry",
                "code": a.get("code", ""),
                "label": a.get("label", ""),
                "vocabulary": a.get("vocabulary_name", vocab),
                "parent_code": a.get("parent_code", code),
                "source": "retrieval_ancestry",
            })

    return results


def _expand_alias_hit(
    engine: RetrievalEngine, hit: dict[str, Any]
) -> list[dict[str, Any]]:
    """Expand an Alias vector hit by dereferencing REFERS_TO target.

    Delegates to the target node type's expansion.
    """
    results: list[dict[str, Any]] = []
    text = hit.get("text", hit.get("name", ""))
    target_name = hit.get("parent_name", hit.get("target_name", ""))

    results.append({
        "type": "alias",
        "text": text,
        "target": target_name,
        "score": hit.get("final_score", 0.0),
        "source": "retrieval",
    })

    # Try to expand target as entity
    if target_name:
        physical = _expand_physical(engine, [target_name])
        for r in physical:
            results.append({
                "type": "entity",
                "name": target_name,
                "table": r.get("table_name", ""),
                "schema": r.get("schema_name", ""),
                "catalog": r.get("catalog", ""),
                "description": None,
                "confidence": 0.5,
                "source": "retrieval_alias_expansion",
                "columns": r.get("columns", []),
            })

    return results
