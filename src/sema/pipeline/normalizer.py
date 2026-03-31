"""AssertionNormalizer: converts extraction DTOs to canonical assertions.

This is the single translation point between connector output and
the assertion store. Connectors emit DTOs; this module emits assertions.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.extraction import (
    ExtractedColumn,
    ExtractedForeignKey,
    ExtractedSampleRows,
    ExtractedTable,
    ExtractedTag,
    ExtractedTopValues,
)


class AssertionNormalizer:
    """Convert extraction DTOs to canonical assertions."""

    def __init__(
        self,
        workspace: str,
        run_id: str,
        platform: str = "databricks",
    ) -> None:
        self._workspace = workspace
        self._run_id = run_id
        self._platform = platform

    def _table_ref(self, catalog: str, schema: str, table: str) -> str:
        return (
            f"{self._platform}://{self._workspace}"
            f"/{catalog}/{schema}/{table}"
        )

    def _column_ref(
        self, catalog: str, schema: str, table: str, column: str,
    ) -> str:
        return f"{self._table_ref(catalog, schema, table)}/{column}"

    def _make(
        self,
        subject_ref: str,
        predicate: AssertionPredicate,
        payload: dict[str, Any],
        source: str = "unity_catalog",
        confidence: float = 0.95,
        object_ref: str | None = None,
    ) -> Assertion:
        return Assertion(
            id=str(uuid.uuid4()),
            subject_ref=subject_ref,
            predicate=predicate,
            payload=payload,
            object_ref=object_ref,
            source=source,
            confidence=confidence,
            run_id=self._run_id,
            observed_at=datetime.now(timezone.utc),
        )

    def normalize_table(self, table: ExtractedTable) -> list[Assertion]:
        ref = self._table_ref(table.catalog, table.schema, table.name)
        assertions = [
            self._make(ref, AssertionPredicate.TABLE_EXISTS, {
                "table_type": "TABLE",
            }),
        ]
        if table.comment:
            assertions.append(
                self._make(ref, AssertionPredicate.HAS_COMMENT, {
                    "value": table.comment,
                })
            )
        return assertions

    def normalize_column(self, col: ExtractedColumn) -> list[Assertion]:
        ref = self._column_ref(
            col.catalog, col.schema, col.table_name, col.name,
        )
        assertions = [
            self._make(ref, AssertionPredicate.COLUMN_EXISTS, {
                "data_type": col.data_type,
                "nullable": col.nullable,
                "comment": col.comment,
            }),
            self._make(ref, AssertionPredicate.HAS_DATATYPE, {
                "value": col.data_type,
            }),
        ]
        if col.comment:
            assertions.append(
                self._make(ref, AssertionPredicate.HAS_COMMENT, {
                    "value": col.comment,
                })
            )
        return assertions

    def normalize_foreign_key(
        self, fk: ExtractedForeignKey,
    ) -> list[Assertion]:
        from_ref = self._table_ref(
            fk.from_catalog, fk.from_schema, fk.from_table,
        )
        to_ref = self._table_ref(
            fk.to_catalog, fk.to_schema, fk.to_table,
        )
        return [
            self._make(
                from_ref,
                AssertionPredicate.JOINS_TO,
                {
                    "on_column": fk.from_columns[0] if fk.from_columns else "",
                    "to_column": fk.to_columns[0] if fk.to_columns else "",
                },
                object_ref=to_ref,
            ),
        ]

    def normalize_top_values(
        self, tv: ExtractedTopValues,
    ) -> list[Assertion]:
        ref = self._column_ref(
            tv.catalog, tv.schema, tv.table_name, tv.column_name,
        )
        return [
            self._make(ref, AssertionPredicate.HAS_TOP_VALUES, {
                "values": tv.values,
                "approx_distinct": tv.approx_distinct,
            }),
        ]

    def normalize_sample_rows(
        self, sr: ExtractedSampleRows,
    ) -> list[Assertion]:
        ref = self._table_ref(sr.catalog, sr.schema, sr.table_name)
        return [
            self._make(ref, AssertionPredicate.HAS_SAMPLE_ROWS, {
                "rows": sr.rows,
                "columns": sr.column_names,
            }),
        ]

    def normalize_tag(self, tag: ExtractedTag) -> list[Assertion]:
        if tag.column_name:
            ref = self._column_ref(
                tag.catalog, tag.schema, tag.table_name, tag.column_name,
            )
        else:
            ref = self._table_ref(
                tag.catalog, tag.schema, tag.table_name,
            )
        return [
            self._make(ref, AssertionPredicate.HAS_TAG, {
                "tag_key": tag.tag_key,
                "tag_value": tag.tag_value,
            }),
        ]
