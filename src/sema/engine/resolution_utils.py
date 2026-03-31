from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)
from sema.models.physical_key import CanonicalRef

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader

logger = logging.getLogger(__name__)


def _parse_ref(ref: str) -> tuple[str, str, str, str | None]:
    try:
        pk = CanonicalRef.parse(ref)
        return pk.catalog_or_db, pk.schema or "", pk.table, pk.column
    except ValueError:
        return "", "", ref, None


def _pick_winner(assertions: list[Assertion]) -> Assertion | None:
    from sema.models.constants import source_precedence

    if not assertions:
        return None

    active = [a for a in assertions
              if a.status not in (AssertionStatus.REJECTED, AssertionStatus.SUPERSEDED)]
    if not active:
        return None

    pinned = [a for a in active if a.status == AssertionStatus.PINNED]
    if pinned:
        return pinned[0]

    accepted = [a for a in active if a.status == AssertionStatus.ACCEPTED]
    if accepted:
        return accepted[0]

    return max(active, key=lambda a: (
        source_precedence(a.source),
        a.confidence,
    ))


def resolve_entities(
    groups: dict[tuple[str, str], list[Assertion]],
    loader: GraphLoader,
) -> None:
    entity_assertions = {
        subj: group
        for (subj, pred), group in groups.items()
        if pred == AssertionPredicate.HAS_ENTITY_NAME.value
    }
    for subject_ref, group in entity_assertions.items():
        winner = _pick_winner(group)
        if not winner:
            continue
        catalog, schema, table, _ = _parse_ref(subject_ref)
        loader.upsert_entity(
            name=winner.payload.get("value", ""),
            description=winner.payload.get("description"),
            source=winner.source,
            confidence=winner.confidence,
            table_name=table,
            schema_name=schema,
            catalog=catalog,
        )


def resolve_properties(
    groups: dict[tuple[str, str], list[Assertion]],
    loader: GraphLoader,
) -> None:
    prop_name_groups = {
        subj: group
        for (subj, pred), group in groups.items()
        if pred == AssertionPredicate.HAS_PROPERTY_NAME.value
    }
    for col_ref, group in prop_name_groups.items():
        winner = _pick_winner(group)
        if not winner:
            continue
        catalog, schema, table, column = _parse_ref(col_ref)
        if not column:
            continue

        type_group = groups.get((col_ref, AssertionPredicate.HAS_SEMANTIC_TYPE.value), [])
        type_winner = _pick_winner(type_group)
        semantic_type = type_winner.payload.get("value", "free_text") if type_winner else "free_text"

        table_ref = f"unity://{catalog}.{schema}.{table}" if catalog else col_ref.rsplit(".", 1)[0]
        entity_group = groups.get((table_ref, AssertionPredicate.HAS_ENTITY_NAME.value), [])
        entity_winner = _pick_winner(entity_group)
        entity_name = entity_winner.payload.get("value", table) if entity_winner else table

        loader.upsert_property(
            name=winner.payload.get("value", ""),
            semantic_type=semantic_type,
            source=winner.source,
            confidence=winner.confidence,
            entity_name=entity_name,
            column_name=column,
            table_name=table,
            schema_name=schema,
            catalog=catalog,
        )


def resolve_decoded_values(
    groups: dict[tuple[str, str], list[Assertion]],
    loader: GraphLoader,
) -> None:
    decoded_groups: dict[str, list[Assertion]] = defaultdict(list)
    for (subj, pred), group in groups.items():
        if pred == AssertionPredicate.HAS_DECODED_VALUE.value:
            decoded_groups[subj].extend(group)

    for col_ref, decoded_assertions in decoded_groups.items():
        catalog, schema, table, column = _parse_ref(col_ref)
        if not column:
            continue

        vs_name = f"{table}_{column}_values"
        loader.upsert_value_set(
            vs_name, column_name=column, table_name=table,
            schema_name=schema, catalog=catalog,
        )

        for a in decoded_assertions:
            if a.status in (AssertionStatus.REJECTED, AssertionStatus.SUPERSEDED):
                continue
            raw = a.payload.get("raw", "")
            label = a.payload.get("label", raw)
            loader.upsert_term(raw, label, source=a.source, confidence=a.confidence)
            loader.add_term_to_value_set(raw, vs_name)


def _collect_alias_groups(
    groups: dict[tuple[str, str], list[Assertion]],
) -> dict[str, list[Assertion]]:
    """Collect alias assertions keyed by subject_ref."""
    alias_groups: dict[str, list[Assertion]] = {}
    for (subj, pred), group in groups.items():
        if pred in (
            AssertionPredicate.HAS_ALIAS.value,
            AssertionPredicate.HAS_SYNONYM.value,  # backward compat
        ):
            alias_groups.setdefault(subj, []).extend(group)
    return alias_groups


def resolve_aliases(
    groups: dict[tuple[str, str], list[Assertion]],
    loader: GraphLoader,
) -> None:
    """Resolve HAS_ALIAS (and deprecated HAS_SYNONYM) assertions into synonym nodes."""
    alias_groups = _collect_alias_groups(groups)
    for subject_ref, group in alias_groups.items():
        _, _, table_or_col, column = _parse_ref(subject_ref)
        if column:
            prop_group = groups.get(
                (subject_ref, AssertionPredicate.HAS_PROPERTY_NAME.value), []
            )
            prop_winner = _pick_winner(prop_group)
            parent_label = ":Property"
            parent_name = (
                prop_winner.payload.get("value", column) if prop_winner else column
            )
        else:
            entity_group = groups.get(
                (subject_ref, AssertionPredicate.HAS_ENTITY_NAME.value), []
            )
            entity_winner = _pick_winner(entity_group)
            parent_label = ":Entity"
            parent_name = (
                entity_winner.payload.get("value", table_or_col)
                if entity_winner
                else table_or_col
            )

        for a in group:
            if a.status in (AssertionStatus.REJECTED, AssertionStatus.SUPERSEDED):
                continue
            loader.upsert_alias(
                text=a.payload.get("value", ""),
                parent_label=parent_label,
                parent_name=parent_name,
                source=a.source,
                confidence=a.confidence,
                is_preferred=a.payload.get("is_preferred", False),
                description=a.payload.get("description"),
            )


# Deprecated alias kept for backward compatibility
resolve_synonyms = resolve_aliases


def resolve_join_paths(
    groups: dict[tuple[str, str], list[Assertion]],
) -> list[dict[str, Any]]:
    """Return join path data structures from HAS_JOIN_EVIDENCE assertions.

    Does NOT call loader methods — returns raw data for callers to act on.
    Each dict has: subject_ref, object_ref, join_predicates, hop_count,
    cardinality, source, confidence.
    """
    results: list[dict[str, Any]] = []
    for (subj, pred), group in groups.items():
        if pred != AssertionPredicate.HAS_JOIN_EVIDENCE.value:
            continue
        for a in group:
            if a.status in (AssertionStatus.REJECTED, AssertionStatus.SUPERSEDED):
                continue
            results.append({
                "subject_ref": a.subject_ref,
                "object_ref": a.object_ref,
                "join_predicates": a.payload.get("join_predicates", []),
                "hop_count": a.payload.get("hop_count", 1),
                "cardinality": a.payload.get("cardinality", "unknown"),
                "source": a.source,
                "confidence": a.confidence,
            })
    return results


def resolve_metrics(
    groups: dict[tuple[str, str], list[Assertion]],
) -> list[dict[str, Any]]:
    """Return metric data structures from metric-related assertions.

    Collects MEASURES, AGGREGATES, FILTERS_BY, AT_GRAIN predicates.
    Does NOT call loader methods — returns raw data for callers to act on.
    Each dict has: subject_ref, predicate, object_ref, payload, source,
    confidence.
    """
    metric_predicates = {
        "measures", "aggregates", "filters_by", "at_grain",
    }
    results: list[dict[str, Any]] = []
    for (subj, pred), group in groups.items():
        if pred not in metric_predicates:
            continue
        for a in group:
            if a.status in (AssertionStatus.REJECTED, AssertionStatus.SUPERSEDED):
                continue
            results.append({
                "subject_ref": a.subject_ref,
                "predicate": pred,
                "object_ref": a.object_ref,
                "payload": a.payload,
                "source": a.source,
                "confidence": a.confidence,
            })
    return results


def resolve_hierarchies(
    groups: dict[tuple[str, str], list[Assertion]],
    loader: GraphLoader,
) -> None:
    for (subj, pred), group in groups.items():
        if pred == AssertionPredicate.PARENT_OF.value:
            for a in group:
                if a.status in (AssertionStatus.REJECTED, AssertionStatus.SUPERSEDED):
                    continue
                loader.add_term_hierarchy(
                    parent_code=a.payload.get("parent", ""),
                    child_code=a.payload.get("child", ""),
                )
