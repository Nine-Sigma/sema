"""Helper functions for the unified materializer.

Extracted from materializer.py to keep that module under 400 lines.
All helpers are private — imported only by materializer.py.
"""

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
    batch_upsert_aliases,
    batch_upsert_entities,
    batch_upsert_properties,
    batch_upsert_terms,
    batch_upsert_value_sets,
)
from sema.models.constants import source_precedence
from sema.models.physical_key import CanonicalRef

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader

logger = logging.getLogger(__name__)


def pick_winner(assertions: list[Assertion]) -> Assertion | None:
    """Select the winning assertion from a group (legacy path).

    Uses assertion.status field directly. The new StatusEvent-based
    winner selection is in sema.models.winner_selection.
    """
    active = [
        a for a in assertions
        if a.status not in (
            AssertionStatus.REJECTED, AssertionStatus.SUPERSEDED,
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
    return max(
        active,
        key=lambda a: (source_precedence(a.source), a.confidence),
    )


def parse_ref_any(ref: str) -> tuple[str, str, str, str | None]:
    """Parse any ref format into (catalog, schema, table, column)."""
    pk = CanonicalRef.parse(ref)
    return pk.catalog_or_db, pk.schema or "", pk.table, pk.column



def upsert_physical_nodes(
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

        catalog, schema, table, _ = parse_ref_any(subject_ref)
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

        table_type = table_exists[0].payload.get("table_type", "TABLE")
        loader.upsert_table(
            table, schema, catalog,
            table_type=table_type, comment=comment, ref=subject_ref,
        )


def upsert_column_nodes(
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

        catalog, schema, table, column = parse_ref_any(subject_ref)
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



def upsert_entities(
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
        winner = pick_winner(group)
        if not winner:
            continue
        catalog, schema, table, _ = parse_ref_any(subject_ref)
        batch.append({
            "name": winner.payload.get("value", ""),
            "description": winner.payload.get("description"),
            "source": winner.source,
            "confidence": winner.confidence,
            "table_name": table,
            "schema_name": schema,
            "catalog": catalog,
        })
    batch_upsert_entities(loader, batch)


def _resolve_property_details(
    col_ref: str,
    group: list[Assertion],
    groups: dict[tuple[str, str], list[Assertion]],
) -> dict[str, Any] | None:
    winner = pick_winner(group)
    if not winner:
        return None
    catalog, schema, table, column = parse_ref_any(col_ref)
    if not column:
        return None

    type_group = groups.get(
        (col_ref, AssertionPredicate.HAS_SEMANTIC_TYPE.value), [],
    )
    type_winner = pick_winner(type_group)
    semantic_type = (
        type_winner.payload.get("value", "free_text")
        if type_winner else "free_text"
    )

    try:
        pk = CanonicalRef.parse(col_ref)
        table_ref = col_ref.rsplit("/", 1)[0] if pk.column else col_ref
    except ValueError:
        table_ref = col_ref.rsplit("/", 1)[0]

    entity_group = groups.get(
        (table_ref, AssertionPredicate.HAS_ENTITY_NAME.value), [],
    )
    entity_winner = pick_winner(entity_group)
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


def upsert_properties(
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
        details = _resolve_property_details(col_ref, group, groups)
        if details:
            batch.append(details)
    batch_upsert_properties(loader, batch)


def upsert_decoded_values(
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
        catalog, schema, table, column = parse_ref_any(col_ref)
        if not column:
            continue
        vs_name = f"{table}_{column}_values"
        vs_batch.append({
            "name": vs_name,
            "column_name": column, "table_name": table,
            "schema_name": schema, "catalog": catalog,
        })
        for a in decoded:
            if a.status in (
                AssertionStatus.REJECTED, AssertionStatus.SUPERSEDED,
            ):
                continue
            raw = a.payload.get("raw", "")
            label = a.payload.get("label", raw)
            term_batch.append({
                "code": raw, "label": label,
                "source": a.source, "confidence": a.confidence,
            })
            loader.add_term_to_value_set(raw, vs_name)

    batch_upsert_value_sets(loader, vs_batch)
    batch_upsert_terms(loader, term_batch)


def _collect_alias_batch(
    subject_ref: str,
    group: list[Assertion],
    groups: dict[tuple[str, str], list[Assertion]],
) -> tuple[list[dict[str, Any]], str]:
    """Collect alias batch entries and determine parent label."""
    _, _, table_or_col, column = parse_ref_any(subject_ref)
    batch: list[dict[str, Any]] = []

    if column:
        prop_group = groups.get(
            (subject_ref, AssertionPredicate.HAS_PROPERTY_NAME.value), [],
        )
        prop_winner = pick_winner(prop_group)
        parent_name = (
            prop_winner.payload.get("value", column)
            if prop_winner else column
        )
        parent_label = ":Property"
    else:
        entity_group = groups.get(
            (subject_ref, AssertionPredicate.HAS_ENTITY_NAME.value), [],
        )
        entity_winner = pick_winner(entity_group)
        parent_name = (
            entity_winner.payload.get("value", table_or_col)
            if entity_winner else table_or_col
        )
        parent_label = ":Entity"

    for a in group:
        if a.status in (
            AssertionStatus.REJECTED, AssertionStatus.SUPERSEDED,
        ):
            continue
        batch.append({
            "text": a.payload.get("value", ""),
            "parent_name": parent_name,
            "source": a.source, "confidence": a.confidence,
            "is_preferred": a.payload.get("is_preferred", False),
            "description": a.payload.get("description"),
        })

    return batch, parent_label


def upsert_aliases(
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
        batch, parent_label = _collect_alias_batch(
            subject_ref, group, groups,
        )
        if parent_label == ":Entity":
            entity_aliases.extend(batch)
        else:
            property_aliases.extend(batch)

    batch_upsert_aliases(loader, entity_aliases, ":Entity")
    batch_upsert_aliases(loader, property_aliases, ":Property")


def upsert_semantic_nodes(
    loader: GraphLoader,
    by_subject: dict[str, list[Assertion]],
    groups: dict[tuple[str, str], list[Assertion]],
) -> None:
    upsert_entities(loader, groups)
    upsert_properties(loader, groups)
    upsert_decoded_values(loader, groups)
    upsert_aliases(loader, groups)
    from sema.graph.join_materializer import materialize_join_paths
    materialize_join_paths(loader, groups)


def apply_resolution_edges(
    loader: GraphLoader,
    groups: dict[tuple[str, str], list[Assertion]],
) -> None:
    for (subj, pred), group in groups.items():
        if pred == AssertionPredicate.PARENT_OF.value:
            for a in group:
                if a.status in (
                    AssertionStatus.REJECTED, AssertionStatus.SUPERSEDED,
                ):
                    continue
                loader.add_term_hierarchy(
                    parent_code=a.payload.get("parent", ""),
                    child_code=a.payload.get("child", ""),
                )



def run_lifecycle_phase(
    loader: GraphLoader,
    assertions: list[Assertion],
) -> None:
    """Deprecate non-anchored nodes that lost all supporting assertions."""
    active_vocabs: set[str] = set()
    for a in assertions:
        if (
            a.predicate == AssertionPredicate.VOCABULARY_MATCH
            and a.status not in (
                AssertionStatus.REJECTED, AssertionStatus.SUPERSEDED,
            )
        ):
            vocab_name = a.payload.get("value")
            if vocab_name:
                active_vocabs.add(vocab_name)

    if active_vocabs:
        loader._run(
            "MATCH (v:Vocabulary) "
            "WHERE v.status = 'ACTIVE' "
            "AND NOT v.name IN $active_names "
            "SET v.status = 'DEPRECATED'",
            active_names=list(active_vocabs),
        )
