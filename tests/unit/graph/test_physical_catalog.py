"""Unit tests for the warehouse `PhysicalCatalogSource` in core."""

from __future__ import annotations

from typing import Any

import pytest

from sema.graph.physical_catalog import DatabricksCatalogSource, PhysicalColumn

pytestmark = pytest.mark.unit


class _FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows
        self.executed: list[tuple[str, list[Any]]] = []

    def execute(self, sql: str, params: list[Any]) -> None:
        self.executed.append((sql, params))

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)


def test_reads_columns_from_information_schema() -> None:
    cursor = _FakeCursor([("doc_id", "STRING", "NO"), ("amount", "DOUBLE", "YES")])
    columns = DatabricksCatalogSource(cursor).columns(
        catalog="cat", schema="sch", table="tbl"
    )
    assert columns == [
        PhysicalColumn("doc_id", "STRING", False),
        PhysicalColumn("amount", "DOUBLE", True),
    ]
    [(sql, params)] = cursor.executed
    assert "cat.information_schema.columns" in sql
    assert "ORDER BY ordinal_position" in sql
    assert params == ["sch", "tbl"]


def test_unknown_table_yields_no_columns() -> None:
    cursor = _FakeCursor([])
    assert DatabricksCatalogSource(cursor).columns(
        catalog="cat", schema="sch", table="missing"
    ) == []


def test_showcase_reuses_the_core_source() -> None:
    from showcase.cbioportal_to_omop import materialize_target_graph

    assert materialize_target_graph.DatabricksCatalogSource is DatabricksCatalogSource
