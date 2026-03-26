from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.constants import parse_ref, parse_unity_ref_strict

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader

logger = logging.getLogger(__name__)


def _parse_ref_parts(ref: str) -> tuple[str, str, str, str | None]:
    """Parse catalog/schema/table/column from either ref format.

    Supports databricks://<workspace>/<catalog>/<schema>/<table>[/<column>]
    and legacy unity://<catalog>.<schema>.<table>[.<column>].
    Returns (catalog, schema, table, column).
    Raises ValueError if ref cannot be parsed.
    """
    parts = parse_ref(ref)
    if parts is not None:
        return parts.catalog, parts.schema, parts.table, parts.column
    # Fall back to legacy unity:// parser
    return parse_unity_ref_strict(ref)


def process_tables(
    loader: GraphLoader,
    by_subject: dict[str, list[Assertion]],
    created_catalogs: set[str],
    created_schemas: set[tuple[str, str]],
) -> None:
    for subject_ref, subject_assertions in by_subject.items():
        table_exists = [a for a in subject_assertions
                      if a.predicate == AssertionPredicate.TABLE_EXISTS]
        if not table_exists:
            continue

        try:
            catalog, schema, table, _ = _parse_ref_parts(subject_ref)
        except ValueError:
            continue

        if catalog not in created_catalogs:
            loader.upsert_catalog(catalog)
            created_catalogs.add(catalog)

        if (schema, catalog) not in created_schemas:
            loader.upsert_schema(schema, catalog)
            created_schemas.add((schema, catalog))

        comment = None
        for a in subject_assertions:
            if a.predicate == AssertionPredicate.HAS_COMMENT:
                comment = a.payload.get("value")

        table_type = table_exists[0].payload.get("table_type", "TABLE")
        loader.upsert_table(table, schema, catalog,
                            table_type=table_type, comment=comment)

        _process_join_assertions(
            loader, subject_assertions, table, schema, catalog
        )


def _process_join_assertions(
    loader: GraphLoader,
    subject_assertions: list[Assertion],
    table: str,
    schema: str,
    catalog: str,
) -> None:
    """Emit join path records for JOINS_TO / HAS_JOIN_EVIDENCE assertions on a table."""
    join_predicates = (
        AssertionPredicate.JOINS_TO,
        AssertionPredicate.HAS_JOIN_EVIDENCE,
    )
    for a in subject_assertions:
        if a.predicate not in join_predicates or not a.object_ref:
            continue
        try:
            to_cat, to_schema, to_table, _ = _parse_ref_parts(a.object_ref)
            join_cols = a.payload.get(
                "join_predicates",
                [a.payload.get("on_column", "")],
            )
            name = (
                f"{catalog}.{schema}.{table}"
                f"--{to_cat}.{to_schema}.{to_table}"
                f"[{','.join(join_cols)}]"
            )
            loader.upsert_join_path(
                name=name,
                join_predicates=[{"column": c} for c in join_cols],
                hop_count=a.payload.get("hop_count", 1),
                source=a.source,
                confidence=a.confidence,
                cardinality_hint=a.payload.get("cardinality", "unknown"),
            )
        except (ValueError, AttributeError):
            logger.warning("Invalid join target ref: %s", a.object_ref)


def process_columns(
    loader: GraphLoader,
    by_subject: dict[str, list[Assertion]],
) -> None:
    for subject_ref, subject_assertions in by_subject.items():
        col_exists = [a for a in subject_assertions
                     if a.predicate == AssertionPredicate.COLUMN_EXISTS]
        if not col_exists:
            continue

        try:
            catalog, schema, table, column = _parse_ref_parts(subject_ref)
        except ValueError:
            continue

        if column is None:
            continue

        col_data = col_exists[0].payload
        loader.upsert_column(
            column, table, schema, catalog,
            data_type=col_data.get("data_type", "UNKNOWN"),
            nullable=col_data.get("nullable", True),
            comment=col_data.get("comment"),
        )
