"""Shared setup for test_intake_extract_hierarchy.py and the tests that reuse it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from showcase.ontology_query_benchmark.intake_extract import extract
from tests.unit.ontology_query_benchmark.hierarchy_fixture import (
    WarehouseCursor,
    hierarchy,
    spec_with,
    write_product_spec,
)


def _extract(tmp_path: Path, cursor: WarehouseCursor, **kwargs: Any) -> dict[str, Any]:
    spec = kwargs.pop("spec", spec_with(hierarchy()))
    return extract(lambda: cursor, spec, [write_product_spec(tmp_path / "events.json")],
                   tmp_path / "out", **kwargs)  # fmt: skip


def _slice(tmp_path: Path, receipt: dict[str, Any], table: str) -> set[tuple[Any, ...]]:
    with duckdb.connect(str(tmp_path / "out" / receipt["duckdb"]["path"]), read_only=True) as con:
        return set(con.execute(f"SELECT * FROM concept_hierarchy.{table}").fetchall())
