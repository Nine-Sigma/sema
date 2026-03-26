from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.constants import parse_unity_ref_strict

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader

logger = logging.getLogger(__name__)


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
            catalog, schema, table, _ = parse_unity_ref_strict(subject_ref)
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

        for a in subject_assertions:
            if a.predicate == AssertionPredicate.JOINS_TO and a.object_ref:
                try:
                    to_cat, to_schema, to_table, _ = parse_unity_ref_strict(a.object_ref)
                    loader.upsert_candidate_join(
                        from_table=table, from_schema=schema, from_catalog=catalog,
                        to_table=to_table, to_schema=to_schema, to_catalog=to_cat,
                        on_column=a.payload.get("on_column", ""),
                        cardinality=a.payload.get("cardinality", "unknown"),
                        source=a.source, confidence=a.confidence,
                    )
                except ValueError:
                    logger.warning(f"Invalid join target ref: {a.object_ref}")


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
            catalog, schema, table, column = parse_unity_ref_strict(subject_ref)
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
