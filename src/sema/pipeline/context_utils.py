"""Helper functions for the context pipeline.

Extracted from context.py to keep the module focused on the
public prune_to_sco entry point.
"""
from __future__ import annotations

from typing import Any

import json

from pydantic import ValidationError

from sema.log import logger
from sema.models.assertions import STATED_GRADES, VERIFIED
from sema.models.context import (
    AncestryTerm,
    ColumnMatch,
    GovernedValue,
    HierarchyAnchor,
    JoinPath,
    JoinPredicate,
    LabelSource,
    PhysicalAsset,
    Provenance,
    ResolvedEntity,
    ResolvedMetric,
    ResolvedProperty,
    SemanticCandidateSet,
    SharedEntity,
)
from sema.models.target.identity import EntityIdentity
from sema.pipeline.dedup_utils import states_name

def _filter_entity_candidates(
    candidate_set: SemanticCandidateSet,
    max_entities: int,
) -> list[dict[str, Any]]:
    """Every entity candidate of the `max_entities` most confident tables, so a table that
    carries two entities spends one slot."""
    ranked = sorted(
        [c for c in candidate_set.candidates if c.get("type") == "entity"],
        key=lambda c: c.get("confidence", 0),
        reverse=True,
    )
    admitted: set[str] = set()
    kept: list[dict[str, Any]] = []
    for c in ranked:
        table = f"{c.get('catalog')}.{c.get('schema')}.{c.get('table')}"
        if table not in admitted and len(admitted) < max_entities:
            admitted.add(table)
        if table in admitted:
            kept.append(c)
    return kept


def _build_entities_and_assets(
    entity_candidates: list[dict[str, Any]],
) -> tuple[list[ResolvedEntity], list[PhysicalAsset]]:
    entities: list[ResolvedEntity] = []
    assets: dict[str, PhysicalAsset] = {}

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
                vocabulary=col.get("vocabulary"),
                provenance=Provenance(
                    source=ec.get("source", "retrieval"),
                    confidence=ec.get("confidence", 0.5),
                ),
                labels=_labels(col),
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
            **_entity_identity(ec),
        ))

        table_key = f"{ec.get('catalog')}.{ec.get('schema')}.{ec.get('table')}"
        columns = [col.get("column") or "" for col in ec.get("columns", []) if col.get("column")]
        asset = assets.get(table_key)
        if asset is None:
            assets[table_key] = PhysicalAsset(
                catalog=ec.get("catalog", ""),
                schema=ec.get("schema", ""),
                table=ec.get("table", ""),
                columns=columns,
                **_asset_identity(ec),
            )
        elif not asset.columns:
            # An entity the identity step created has no properties; another on its table has.
            asset.columns = columns

    return entities, list(assets.values())


def _entity_identity(ec: dict[str, Any]) -> dict[str, Any]:
    """The entity's key, its identity when coarser than a row, and its NULL-key row count,
    when the edge's grade is one the context states."""
    if ec.get("verification") not in STATED_GRADES:
        return {}
    return {
        "key_columns": list(ec.get("key_columns") or []),
        "identity": _parse_identity(ec.get("identity")),
        "null_key_rows": ec.get("null_key_rows"),
    }


def _parse_identity(raw: Any) -> EntityIdentity | None:
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            return EntityIdentity.model_validate_json(raw)
        return EntityIdentity.model_validate(raw)
    except ValidationError as e:
        logger.warning("Unreadable entity identity {!r}: {}", raw, e)
        return None


def _asset_identity(ec: dict[str, Any]) -> dict[str, Any]:
    """The relation's key when its grade is stated, and the entities it shares."""
    graded = ec.get("table_key_verification") in STATED_GRADES
    return {
        "key_columns": list(ec.get("table_key_columns") or []) if graded else [],
        "shared_entities": [
            SharedEntity(
                entity=s["entity"],
                owner_table=f"{o.get('catalog') or ''}.{o.get('schema') or ''}.{o['table']}",
                key_columns=list(s["key_columns"]),
                identity=_parse_identity(s.get("identity")),
            )
            for s in ec.get("shared_entities") or []
            if s.get("verification") in STATED_GRADES and (o := s.get("owner"))
        ],
    }


def _labels(col: dict[str, Any]) -> list[LabelSource]:
    """A column's verified labels, each with the predicates of the joins its edge names."""
    return [
        LabelSource(
            table=label["table"], column=label["column"],
            via=[p for raw in label.get("via") or [] for p in _parse_join_predicates(raw)],
        )
        for label in col.get("labels") or []
        if label.get("verification") == VERIFIED
    ]


def _build_column_matches(
    candidate_set: SemanticCandidateSet,
    physical_assets: list[PhysicalAsset],
) -> list[ColumnMatch]:
    """Each listed column the question names, paired with every other listed column a
    property hit reached whose definition states that same name. A column the question
    names is never another's alternative."""
    listed = {(a.table, c) for a in physical_assets for c in a.columns}
    hits = {
        place: c for c in candidate_set.candidates
        if c.get("type") == "property"
        and (place := (c.get("physical_table", ""), c.get("physical_column", ""))) in listed
    }
    named = {
        place: term for place, c in hits.items()
        if (term := _named_term(candidate_set.query, c.get("name", ""), place[1]))
    }
    return [
        ColumnMatch(table=table, column=column,
                    alternative_table=other[0], alternative_column=other[1])
        for (table, column), term in named.items()
        for other, c in hits.items()
        if other not in named and states_name(c.get("description") or "", term)
    ]


def _named_term(question: str, *names: str) -> str:
    """The first of ``names`` the question states, or ''."""
    return next((n for n in names if states_name(question, n)), "")


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
        predicates = _parse_join_predicates(jc.get("join_predicates"))
        join_paths.append(JoinPath(
            from_table=predicates[0].left_table if predicates else "",
            to_table=predicates[0].right_table if predicates else "",
            join_predicates=predicates,
            hop_count=jc.get("hop_count", 1),
            cardinality_hint=jc.get("cardinality_hint"),
            sql_snippet=jc.get("sql_snippet"),
            confidence=jc.get("confidence", 0.5),
        ))
    return join_paths


_STRUCTURAL_CONFIDENCE_THRESHOLD = 0.5
_SEMANTIC_CONFIDENCE_THRESHOLD = 0.7


def _passes_confidence_threshold(candidate: dict[str, Any]) -> bool:
    """Return True if an 'auto' candidate meets its source threshold.

    Uses confidence_policy when present, falls back to source-string
    inference when absent.
    """
    confidence = candidate.get("confidence", 0.0)
    policy = candidate.get("confidence_policy")
    if policy is not None:
        threshold = (
            _STRUCTURAL_CONFIDENCE_THRESHOLD
            if policy == "structural"
            else _SEMANTIC_CONFIDENCE_THRESHOLD
        )
    else:
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
            vocabulary=c.get("vocabulary"),
        )
        for c in ancestry_candidates
    ]


def _build_hierarchy_anchors(
    candidate_set: SemanticCandidateSet,
) -> list[HierarchyAnchor]:
    return [
        HierarchyAnchor(
            code=c.get("code", ""),
            label=c.get("label", ""),
            vocabulary=c.get("vocabulary"),
            table=c.get("table", ""),
            column=c.get("column", ""),
            schema_name=c.get("schema", ""),
            member_count=c.get("member_count", 0),
            member_codes=c.get("member_codes", []),
            score=c.get("score", 0.0),
        )
        for c in candidate_set.candidates
        if c.get("type") == "hierarchy_anchor"
    ]


def _build_metrics(
    candidate_set: SemanticCandidateSet,
) -> list[ResolvedMetric]:
    """Build ResolvedMetric list from metric candidates."""
    metric_candidates = [
        c for c in candidate_set.candidates
        if c.get("type") == "metric"
    ]
    return [
        ResolvedMetric(
            name=mc.get("name", ""),
            description=mc.get("description"),
            formula=mc.get("formula"),
            aggregates=mc.get("aggregates", []),
            filters=mc.get("filters", []),
            grains=mc.get("grains", []),
            provenance=Provenance(
                source=mc.get("source", "retrieval"),
                confidence=mc.get("confidence", 0.5),
            ),
        )
        for mc in metric_candidates
    ]


def _build_governed_values(
    candidate_set: SemanticCandidateSet,
) -> list[GovernedValue]:
    """Group value candidates per column, keeping vocabulary and
    ambiguity apart so alternatives never merge into one value list."""
    value_candidates = [
        c for c in candidate_set.candidates if c.get("type") == "value"
    ]
    GroupKey = tuple[str, str, str, str, bool]
    value_groups: dict[GroupKey, list[dict[str, str | None]]] = {}
    for vc in value_candidates:
        key = (
            vc.get("property_name", ""),
            vc.get("column", ""),
            vc.get("table", ""),
            vc.get("vocabulary") or "",
            bool(vc.get("ambiguous")),
        )
        value_groups.setdefault(key, []).append(
            {"code": vc.get("code", ""), "label": vc.get("label", "")}
        )

    governed_values: list[GovernedValue] = []
    for (prop_name, col, tbl, vocab, ambiguous), values in (
        value_groups.items()
    ):
        governed_values.append(GovernedValue(
            property_name=prop_name,
            column=col,
            table=tbl,
            values=values,
            vocabulary=vocab or None,
            ambiguous=ambiguous,
        ))
    return governed_values
