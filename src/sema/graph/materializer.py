from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)
from sema.graph.loader_utils import (
    batch_upsert_entities as _batch_upsert_entities,
    batch_upsert_properties as _batch_upsert_properties,
    batch_upsert_synonyms as _batch_upsert_synonyms,
    batch_upsert_terms as _batch_upsert_terms,
    batch_upsert_value_sets as _batch_upsert_value_sets,
)
from sema.models.constants import (
    parse_unity_ref,
    source_precedence,
)

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader

logger = logging.getLogger(__name__)


def _pick_winner(assertions: list[Assertion]) -> Assertion | None:
    active = [
        a for a in assertions
        if a.status not in (
            AssertionStatus.REJECTED, AssertionStatus.SUPERSEDED
        )
    ]
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


def _upsert_physical_nodes(
    loader: GraphLoader,
    by_subject: dict[str, list[Assertion]],
) -> None:
    created_catalogs: set[str] = set()
    created_schemas: set[tuple[str, str]] = set()

    for subject_ref, subj_assertions in by_subject.items():
        table_exists = [
            a for a in subj_assertions
            if a.predicate == AssertionPredicate.TABLE_EXISTS
        ]
        if not table_exists:
            continue

        catalog, schema, table, _ = parse_unity_ref(subject_ref)
        if not catalog:
            continue

        if catalog not in created_catalogs:
            loader.upsert_catalog(catalog)
            created_catalogs.add(catalog)

        if (schema, catalog) not in created_schemas:
            loader.upsert_schema(schema, catalog)
            created_schemas.add((schema, catalog))

        comment = None
        for a in subj_assertions:
            if a.predicate == AssertionPredicate.HAS_COMMENT:
                comment = a.payload.get("value")

        table_type = table_exists[0].payload.get(
            "table_type", "TABLE"
        )
        loader.upsert_table(
            table, schema, catalog,
            table_type=table_type, comment=comment,
        )

        for a in subj_assertions:
            if (
                a.predicate == AssertionPredicate.JOINS_TO
                and a.object_ref
            ):
                to_cat, to_sch, to_tbl, _ = parse_unity_ref(a.object_ref)
                if to_cat:
                    loader.upsert_candidate_join(
                        from_table=table,
                        from_schema=schema,
                        from_catalog=catalog,
                        to_table=to_tbl,
                        to_schema=to_sch,
                        to_catalog=to_cat,
                        on_column=a.payload.get("on_column", ""),
                        cardinality=a.payload.get(
                            "cardinality", "unknown"
                        ),
                        source=a.source,
                        confidence=a.confidence,
                    )


def _upsert_column_nodes(
    loader: GraphLoader,
    by_subject: dict[str, list[Assertion]],
) -> None:
    for subject_ref, subj_assertions in by_subject.items():
        col_exists = [
            a for a in subj_assertions
            if a.predicate == AssertionPredicate.COLUMN_EXISTS
        ]
        if not col_exists:
            continue

        catalog, schema, table, column = parse_unity_ref(subject_ref)
        if not column:
            continue

        col_data = col_exists[0].payload
        loader.upsert_column(
            column, table, schema, catalog,
            data_type=col_data.get("data_type", "UNKNOWN"),
            nullable=col_data.get("nullable", True),
            comment=col_data.get("comment"),
        )


def _upsert_entities(
    loader: GraphLoader,
    groups: dict[tuple[str, str], list[Assertion]],
) -> None:
    entity_groups = {
        subj: group
        for (subj, pred), group in groups.items()
        if pred == AssertionPredicate.HAS_ENTITY_NAME.value
    }
    batch: list[dict[str, Any]] = []
    for subject_ref, group in entity_groups.items():
        winner = _pick_winner(group)
        if not winner:
            continue
        catalog, schema, table, _ = parse_unity_ref(subject_ref)
        batch.append({
            "name": winner.payload.get("value", ""),
            "description": winner.payload.get("description"),
            "source": winner.source,
            "confidence": winner.confidence,
            "table_name": table,
            "schema_name": schema,
            "catalog": catalog,
        })
    _batch_upsert_entities(loader, batch)


def _resolve_property_details(
    col_ref: str,
    group: list[Assertion],
    groups: dict[tuple[str, str], list[Assertion]],
) -> dict[str, Any] | None:
    winner = _pick_winner(group)
    if not winner:
        return None
    catalog, schema, table, column = parse_unity_ref(col_ref)
    if not column:
        return None

    type_group = groups.get(
        (col_ref, AssertionPredicate.HAS_SEMANTIC_TYPE.value),
        [],
    )
    type_winner = _pick_winner(type_group)
    semantic_type = (
        type_winner.payload.get("value", "free_text")
        if type_winner
        else "free_text"
    )

    table_ref = (
        f"unity://{catalog}.{schema}.{table}"
        if catalog
        else col_ref.rsplit(".", 1)[0]
    )
    entity_group = groups.get(
        (table_ref, AssertionPredicate.HAS_ENTITY_NAME.value),
        [],
    )
    entity_winner = _pick_winner(entity_group)
    entity_name = (
        entity_winner.payload.get("value", table)
        if entity_winner
        else table
    )

    return {
        "name": winner.payload.get("value", ""),
        "semantic_type": semantic_type,
        "source": winner.source,
        "confidence": winner.confidence,
        "entity_name": entity_name,
        "column_name": column,
        "table_name": table,
        "schema_name": schema,
        "catalog": catalog,
    }


def _upsert_properties(
    loader: GraphLoader,
    groups: dict[tuple[str, str], list[Assertion]],
) -> None:
    prop_groups = {
        subj: group
        for (subj, pred), group in groups.items()
        if pred == AssertionPredicate.HAS_PROPERTY_NAME.value
    }
    batch: list[dict[str, Any]] = []
    for col_ref, group in prop_groups.items():
        details = _resolve_property_details(
            col_ref, group, groups
        )
        if details:
            batch.append(details)
    _batch_upsert_properties(loader, batch)


def _upsert_decoded_values(
    loader: GraphLoader,
    groups: dict[tuple[str, str], list[Assertion]],
) -> None:
    decoded_groups: dict[str, list[Assertion]] = defaultdict(list)
    for (subj, pred), group in groups.items():
        if pred == AssertionPredicate.HAS_DECODED_VALUE.value:
            decoded_groups[subj].extend(group)

    vs_batch: list[dict[str, Any]] = []
    term_batch: list[dict[str, Any]] = []

    for col_ref, decoded in decoded_groups.items():
        catalog, schema, table, column = parse_unity_ref(col_ref)
        if not column:
            continue
        vs_name = f"{table}_{column}_values"
        vs_batch.append({
            "name": vs_name,
            "column_name": column,
            "table_name": table,
            "schema_name": schema,
            "catalog": catalog,
        })
        for a in decoded:
            if a.status in (
                AssertionStatus.REJECTED,
                AssertionStatus.SUPERSEDED,
            ):
                continue
            raw = a.payload.get("raw", "")
            label = a.payload.get("label", raw)
            term_batch.append({
                "code": raw,
                "label": label,
                "source": a.source,
                "confidence": a.confidence,
            })
            loader.add_term_to_value_set(raw, vs_name)

    _batch_upsert_value_sets(loader, vs_batch)
    _batch_upsert_terms(loader, term_batch)


def _upsert_synonyms(
    loader: GraphLoader,
    groups: dict[tuple[str, str], list[Assertion]],
) -> None:
    synonym_groups = {
        subj: group
        for (subj, pred), group in groups.items()
        if pred == AssertionPredicate.HAS_SYNONYM.value
    }
    entity_synonyms: list[dict[str, Any]] = []
    property_synonyms: list[dict[str, Any]] = []

    for subject_ref, group in synonym_groups.items():
        _, _, table_or_col, column = parse_unity_ref(subject_ref)
        if column:
            prop_group = groups.get(
                (
                    subject_ref,
                    AssertionPredicate.HAS_PROPERTY_NAME.value,
                ),
                [],
            )
            prop_winner = _pick_winner(prop_group)
            parent_name = (
                prop_winner.payload.get("value", column)
                if prop_winner
                else column
            )
            target_list = property_synonyms
        else:
            entity_group = groups.get(
                (
                    subject_ref,
                    AssertionPredicate.HAS_ENTITY_NAME.value,
                ),
                [],
            )
            entity_winner = _pick_winner(entity_group)
            parent_name = (
                entity_winner.payload.get("value", table_or_col)
                if entity_winner
                else table_or_col
            )
            target_list = entity_synonyms

        for a in group:
            if a.status in (
                AssertionStatus.REJECTED,
                AssertionStatus.SUPERSEDED,
            ):
                continue
            target_list.append({
                "text": a.payload.get("value", ""),
                "parent_name": parent_name,
                "source": a.source,
                "confidence": a.confidence,
            })

    _batch_upsert_synonyms(loader, entity_synonyms, ":Entity")
    _batch_upsert_synonyms(loader, property_synonyms, ":Property")


def _upsert_semantic_nodes(
    loader: GraphLoader,
    by_subject: dict[str, list[Assertion]],
    groups: dict[tuple[str, str], list[Assertion]],
) -> None:
    _upsert_entities(loader, groups)
    _upsert_properties(loader, groups)
    _upsert_decoded_values(loader, groups)
    _upsert_synonyms(loader, groups)


def _apply_resolution_edges(
    loader: GraphLoader,
    groups: dict[tuple[str, str], list[Assertion]],
) -> None:
    for (subj, pred), group in groups.items():
        if pred == AssertionPredicate.PARENT_OF.value:
            for a in group:
                if a.status in (
                    AssertionStatus.REJECTED,
                    AssertionStatus.SUPERSEDED,
                ):
                    continue
                loader.add_term_hierarchy(
                    parent_code=a.payload.get("parent", ""),
                    child_code=a.payload.get("child", ""),
                )
