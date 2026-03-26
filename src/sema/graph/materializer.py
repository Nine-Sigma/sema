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
    batch_upsert_aliases as _batch_upsert_aliases,
    batch_upsert_entities as _batch_upsert_entities,
    batch_upsert_join_paths as _batch_upsert_join_paths,
    batch_upsert_properties as _batch_upsert_properties,
    batch_upsert_terms as _batch_upsert_terms,
    batch_upsert_value_sets as _batch_upsert_value_sets,
)
from sema.models.constants import (
    parse_ref,
    parse_unity_ref,
    source_precedence,
)

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader

logger = logging.getLogger(__name__)

PROVENANCE_PREDICATES = frozenset({
    "has_entity_name", "has_property_name", "has_alias",
    "has_semantic_type", "has_decoded_value",
    "vocabulary_match", "parent_of", "has_join_evidence",
})


def _pick_winner(
    assertions: list[Assertion],
) -> Assertion | None:
    active = [
        a for a in assertions
        if a.status not in (
            AssertionStatus.REJECTED,
            AssertionStatus.SUPERSEDED,
        )
    ]
    if not active:
        return None
    pinned = [
        a for a in active if a.status == AssertionStatus.PINNED
    ]
    if pinned:
        return pinned[0]
    accepted = [
        a for a in active
        if a.status == AssertionStatus.ACCEPTED
    ]
    if accepted:
        return accepted[0]
    return max(
        active,
        key=lambda a: (source_precedence(a.source), a.confidence),
    )


def _parse_ref_any(ref: str) -> tuple[str, str, str, str | None]:
    parts = parse_ref(ref)
    if parts:
        return parts.catalog, parts.schema, parts.table, parts.column
    return parse_unity_ref(ref)


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

        catalog, schema, table, _ = _parse_ref_any(subject_ref)
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
            ref=subject_ref,
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

        catalog, schema, table, column = _parse_ref_any(
            subject_ref
        )
        if not column:
            continue

        col_data = col_exists[0].payload
        loader.upsert_column(
            column, table, schema, catalog,
            data_type=col_data.get("data_type", "UNKNOWN"),
            nullable=col_data.get("nullable", True),
            comment=col_data.get("comment"),
            ref=subject_ref,
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
        catalog, schema, table, _ = _parse_ref_any(subject_ref)
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
    catalog, schema, table, column = _parse_ref_any(col_ref)
    if not column:
        return None

    type_group = groups.get(
        (col_ref, AssertionPredicate.HAS_SEMANTIC_TYPE.value), [],
    )
    type_winner = _pick_winner(type_group)
    semantic_type = (
        type_winner.payload.get("value", "free_text")
        if type_winner else "free_text"
    )

    parts = parse_ref(col_ref)
    if parts:
        table_ref = (
            f"{parts.platform}://{parts.workspace}/"
            f"{parts.catalog}/{parts.schema}/{parts.table}"
        )
    else:
        table_ref = (
            f"unity://{catalog}.{schema}.{table}"
            if catalog else col_ref.rsplit(".", 1)[0]
        )

    entity_group = groups.get(
        (table_ref, AssertionPredicate.HAS_ENTITY_NAME.value), [],
    )
    entity_winner = _pick_winner(entity_group)
    entity_name = (
        entity_winner.payload.get("value", table)
        if entity_winner else table
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
            col_ref, group, groups,
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
        catalog, schema, table, column = _parse_ref_any(col_ref)
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


def _upsert_aliases(
    loader: GraphLoader,
    groups: dict[tuple[str, str], list[Assertion]],
) -> None:
    alias_groups = {
        subj: group
        for (subj, pred), group in groups.items()
        if pred in (
            AssertionPredicate.HAS_ALIAS.value,
            AssertionPredicate.HAS_SYNONYM.value,
        )
    }
    entity_aliases: list[dict[str, Any]] = []
    property_aliases: list[dict[str, Any]] = []

    for subject_ref, group in alias_groups.items():
        _, _, table_or_col, column = _parse_ref_any(subject_ref)
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
                if prop_winner else column
            )
            target_list = property_aliases
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
                if entity_winner else table_or_col
            )
            target_list = entity_aliases

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
                "is_preferred": a.payload.get(
                    "is_preferred", False
                ),
                "description": a.payload.get("description"),
            })

    _batch_upsert_aliases(loader, entity_aliases, ":Entity")
    _batch_upsert_aliases(loader, property_aliases, ":Property")


def _build_join_path_records(
    join_groups: dict[str, list[Assertion]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _subject_ref, group in join_groups.items():
        winner = _pick_winner(group)
        if not winner:
            continue
        join_predicates = winner.payload.get("join_predicates", [])
        hop_count = winner.payload.get("hop_count", 1)
        cardinality = winner.payload.get("cardinality")
        name = _derive_join_path_name(join_predicates)
        records.append({
            "name": name,
            "join_predicates": join_predicates,
            "hop_count": hop_count,
            "source": winner.source,
            "confidence": winner.confidence,
            "sql_snippet": winner.payload.get("sql_snippet"),
            "cardinality_hint": cardinality,
            "from_table": winner.payload.get("from_table", ""),
            "to_table": winner.payload.get("to_table", ""),
        })
    return records


def _wire_join_path_edges(
    loader: GraphLoader,
    records: list[dict[str, Any]],
) -> None:
    for rec in records:
        name = rec["name"]
        for jp in rec["join_predicates"]:
            if jp.get("left_table"):
                loader.add_join_path_uses(name, jp["left_table"])
            if jp.get("left_column") and jp.get("left_table"):
                loader.add_join_path_uses(
                    name, jp["left_table"], jp["left_column"]
                )
            if jp.get("right_table"):
                loader.add_join_path_uses(name, jp["right_table"])
            if jp.get("right_column") and jp.get("right_table"):
                loader.add_join_path_uses(
                    name, jp["right_table"], jp["right_column"]
                )
        if rec.get("from_table") or rec.get("to_table"):
            loader.add_join_path_entity_links(
                name,
                rec.get("from_table", ""),
                rec.get("to_table", ""),
            )


def _materialize_join_paths(
    loader: GraphLoader,
    groups: dict[tuple[str, str], list[Assertion]],
) -> None:
    """Create JoinPath nodes then wire USES/FROM_ENTITY/TO_ENTITY edges."""
    join_groups = {
        subj: group
        for (subj, pred), group in groups.items()
        if pred == AssertionPredicate.HAS_JOIN_EVIDENCE.value
    }
    records = _build_join_path_records(join_groups)
    batch = [
        {k: v for k, v in r.items() if k not in ("from_table", "to_table")}
        for r in records
    ]
    _batch_upsert_join_paths(loader, batch)
    _wire_join_path_edges(loader, records)


def _derive_join_path_name(
    join_predicates: list[dict[str, str]],
) -> str:
    parts = []
    for jp in join_predicates:
        left = f"{jp['left_table']}/{jp['left_column']}"
        right = f"{jp['right_table']}/{jp['right_column']}"
        parts.append(f"{left}={right}")
    return "|".join(parts)


def _materialize_provenance_edges(
    loader: GraphLoader,
    assertions: list[Assertion],
) -> None:
    loader.materialize_provenance_edges(assertions)


def _materialize_metrics(
    loader: GraphLoader,
    groups: dict[tuple[str, str], list[Assertion]],
) -> None:
    """Stub: metric materialization wired in future."""
    pass


def _materialize_transformations(
    loader: GraphLoader,
    groups: dict[tuple[str, str], list[Assertion]],
) -> None:
    raise NotImplementedError(
        "Transformation materialization requires a connector "
        "that emits lineage data (dbt, Airflow, Databricks "
        "Workflows). No current connector supports this. "
        "See docs/architecture/graph_data_model_v1.md."
    )


def _cleanup_stale_nodes(
    loader: GraphLoader,
    table_ref: str,
    current_entity_names: set[str],
    current_property_keys: set[tuple[str, str]],
) -> None:
    pass


def _upsert_semantic_nodes(
    loader: GraphLoader,
    by_subject: dict[str, list[Assertion]],
    groups: dict[tuple[str, str], list[Assertion]],
) -> None:
    _upsert_entities(loader, groups)
    _upsert_properties(loader, groups)
    _upsert_decoded_values(loader, groups)
    _upsert_aliases(loader, groups)
    _materialize_join_paths(loader, groups)


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
