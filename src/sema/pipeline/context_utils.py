"""Helper functions for the context pipeline.

Extracted from context.py to keep the module focused on the
public prune_to_sco entry point.
"""
from __future__ import annotations

from typing import Any

from sema.models.context import (
    GovernedValue,
    JoinPath,
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
            properties.append(ResolvedProperty(
                name=col.get("property", col.get("column", "")),
                semantic_type=col.get("semantic_type", "free_text"),
                physical_column=col.get("column", ""),
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
                    col.get("column", "")
                    for col in ec.get("columns", [])
                ],
            ))

    return entities, physical_assets


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
            from_table=f"{jc.get('from_catalog', '')}.{jc.get('from_schema', '')}.{jc.get('from_table', '')}",
            to_table=f"{jc.get('to_catalog', '')}.{jc.get('to_schema', '')}.{jc.get('to_table', '')}",
            on_column=jc.get("on_column", ""),
            cardinality=jc.get("cardinality", "unknown"),
            confidence=jc.get("confidence", 0.5),
        ))
    return join_paths


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
