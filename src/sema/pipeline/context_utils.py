"""Helper functions for the context pipeline.

Extracted from context.py to keep the module focused on the
public prune_to_sco entry point.
"""
from __future__ import annotations

from typing import Any

import json

from sema.models.context import (
    AncestryTerm,
    GovernedValue,
    JoinPath,
    JoinPredicate,
    PhysicalAsset,
    Provenance,
    ResolvedEntity,
    ResolvedProperty,
    SemanticCandidateSet,
)


def _filter_entity_candidates(
    candidate_set: SemanticCandidateSet,
    max_entities: int,
) -> list[dict[str, Any]]:
    return sorted(
        [c for c in candidate_set.candidates if c.get("type") == "entity"],
        key=lambda c: c.get("confidence", 0),
        reverse=True,
    )[:max_entities]


def _build_entities_and_assets(
    entity_candidates: list[dict[str, Any]],
) -> tuple[list[ResolvedEntity], list[PhysicalAsset]]:
    entities: list[ResolvedEntity] = []
    physical_assets: list[PhysicalAsset] = []
    seen_tables: set[str] = set()

    for ec in entity_candidates:
        properties = []
        for col in ec.get("columns", []):
            col_name = col.get("column") or col.get("property") or ""
            if not col_name:
                continue
            properties.append(ResolvedProperty(
                name=col.get("property") or col_name,
                semantic_type=col.get("semantic_type") or "free_text",
                physical_column=col_name,
                physical_table=f"{ec.get('catalog', '')}.{ec.get('schema', '')}.{ec.get('table', '')}",
                provenance=Provenance(
                    source=ec.get("source", "retrieval"),
                    confidence=ec.get("confidence", 0.5),
                ),
            ))

        entities.append(ResolvedEntity(
            name=ec.get("name", ""),
            description=ec.get("description"),
            physical_table=f"{ec.get('catalog', '')}.{ec.get('schema', '')}.{ec.get('table', '')}",
            properties=properties,
            provenance=Provenance(
                source=ec.get("source", "retrieval"),
                confidence=ec.get("confidence", 0.5),
            ),
        ))

        table_key = f"{ec.get('catalog')}.{ec.get('schema')}.{ec.get('table')}"
        if table_key not in seen_tables:
            seen_tables.add(table_key)
            physical_assets.append(PhysicalAsset(
                catalog=ec.get("catalog", ""),
                schema=ec.get("schema", ""),
                table=ec.get("table", ""),
                columns=[
                    col.get("column") or ""
                    for col in ec.get("columns", [])
                    if col.get("column")
                ],
            ))

    return entities, physical_assets


def _parse_join_predicates(
    raw: Any,
) -> list[JoinPredicate]:
    """Parse join_predicates from a graph node (JSON string or list)."""
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    predicates: list[JoinPredicate] = []
    for item in raw:
        if isinstance(item, dict):
            predicates.append(JoinPredicate(
                left_table=item.get("left_table", ""),
                left_column=item.get("left_column", ""),
                right_table=item.get("right_table", ""),
                right_column=item.get("right_column", ""),
                operator=item.get("operator", "="),
            ))
    return predicates


def _build_join_paths(
    candidate_set: SemanticCandidateSet,
    max_joins: int,
) -> list[JoinPath]:
    join_candidates = [
        c for c in candidate_set.candidates if c.get("type") == "join"
    ][:max_joins]
    join_paths: list[JoinPath] = []
    for jc in join_candidates:
        join_paths.append(JoinPath(
            from_table=jc.get("from_table", ""),
            to_table=jc.get("to_table", ""),
            join_predicates=_parse_join_predicates(
                jc.get("join_predicates")
            ),
            hop_count=jc.get("hop_count", 1),
            cardinality_hint=jc.get("cardinality_hint"),
            sql_snippet=jc.get("sql_snippet"),
            confidence=jc.get("confidence", 0.5),
        ))
    return join_paths


_STRUCTURAL_CONFIDENCE_THRESHOLD = 0.5
_SEMANTIC_CONFIDENCE_THRESHOLD = 0.7


def _passes_confidence_threshold(candidate: dict[str, Any]) -> bool:
    """Return True if an 'auto' candidate meets its source threshold."""
    confidence = candidate.get("confidence", 0.0)
    source = candidate.get("source", "")
    threshold = (
        _STRUCTURAL_CONFIDENCE_THRESHOLD
        if source == "structural"
        else _SEMANTIC_CONFIDENCE_THRESHOLD
    )
    return bool(confidence >= threshold)


def _apply_visibility_policy(
    candidates: list[dict[str, Any]],
    consumer: str,
) -> list[dict[str, Any]]:
    """Filter candidates by assertion status.

    - pinned/accepted: always include
    - auto: include if confidence >= threshold (structural: 0.5, semantic: 0.7)
    - rejected/superseded: never include
    """
    result: list[dict[str, Any]] = []
    for c in candidates:
        status = c.get("status", "auto")
        if status in ("rejected", "superseded"):
            continue
        if status in ("pinned", "accepted"):
            result.append(c)
            continue
        if _passes_confidence_threshold(c):
            result.append(c)
    return result


def _build_ancestry(
    candidate_set: SemanticCandidateSet,
) -> list[AncestryTerm]:
    """Build AncestryTerm list from ancestry candidates."""
    ancestry_candidates = [
        c for c in candidate_set.candidates if c.get("type") == "ancestry"
    ]
    return [
        AncestryTerm(
            code=c.get("code", ""),
            label=c.get("label", ""),
            parent_code=c.get("parent_code"),
        )
        for c in ancestry_candidates
    ]


def _build_governed_values(
    candidate_set: SemanticCandidateSet,
) -> list[GovernedValue]:
    value_candidates = [
        c for c in candidate_set.candidates if c.get("type") == "value"
    ]
    value_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for vc in value_candidates:
        key = (
            vc.get("property_name", ""),
            vc.get("column", ""),
            vc.get("table", ""),
        )
        value_groups.setdefault(key, []).append(
            {"code": vc.get("code", ""), "label": vc.get("label", "")}
        )

    governed_values: list[GovernedValue] = []
    for (prop_name, col, tbl), values in value_groups.items():
        governed_values.append(GovernedValue(
            property_name=prop_name,
            column=col,
            table=tbl,
            values=values,
        ))
    return governed_values
