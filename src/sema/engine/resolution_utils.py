from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)
from sema.models.constants import parse_unity_ref

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader

logger = logging.getLogger(__name__)


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
        catalog, schema, table, _ = parse_unity_ref(subject_ref)
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
        catalog, schema, table, column = parse_unity_ref(col_ref)
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
        catalog, schema, table, column = parse_unity_ref(col_ref)
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


def resolve_synonyms(
    groups: dict[tuple[str, str], list[Assertion]],
    loader: GraphLoader,
) -> None:
    synonym_groups = {
        subj: group
        for (subj, pred), group in groups.items()
        if pred == AssertionPredicate.HAS_SYNONYM.value
    }
    for subject_ref, group in synonym_groups.items():
        _, _, table_or_col, column = parse_unity_ref(subject_ref)
        if column:
            # Column-level synonym -> Property
            prop_group = groups.get((subject_ref, AssertionPredicate.HAS_PROPERTY_NAME.value), [])
            prop_winner = _pick_winner(prop_group)
            parent_label = ":Property"
            parent_name = prop_winner.payload.get("value", column) if prop_winner else column
        else:
            # Table-level synonym -> Entity
            entity_group = groups.get((subject_ref, AssertionPredicate.HAS_ENTITY_NAME.value), [])
            entity_winner = _pick_winner(entity_group)
            parent_label = ":Entity"
            parent_name = entity_winner.payload.get("value", table_or_col) if entity_winner else table_or_col

        for a in group:
            if a.status in (AssertionStatus.REJECTED, AssertionStatus.SUPERSEDED):
                continue
            loader.upsert_synonym(
                text=a.payload.get("value", ""),
                parent_label=parent_label,
                parent_name=parent_name,
                source=a.source,
                confidence=a.confidence,
            )


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
