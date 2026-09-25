"""Read-time evidence policy guard (design 5.3, 7.1 check 3 second half).

The graph may have been built earlier under ``full``, so Column, Property and
Table nodes can carry profiled values. Source evidence must contain none of
them. Expected green before and after the slice: the builder never reads them.
"""

from __future__ import annotations

from typing import Any

import pytest

from sema.binding.models import DiscoveryScope
from sema.graph.queries import CypherQueries
from tests.unit.binding.scope_fixture import (
    CATALOG,
    SCHEMA,
    FakeDriver,
    default_graph,
)

pytestmark = pytest.mark.unit

TOP_VALUE = "zz_top_value_sentinel"
SAMPLE_VALUE = "zz_sample_row_sentinel"
VALUE_KEYS = ("top_values", "sample_rows")


def _top_values() -> list[dict[str, Any]]:
    return [{"value": TOP_VALUE, "frequency": 41}]


def _sample_rows() -> list[dict[str, Any]]:
    return [{"race_label": SAMPLE_VALUE, "encounter_id": SAMPLE_VALUE}]


class RichEvidenceDriver(FakeDriver):
    """Scope rows whose nodes carry values a ``full`` build could have written."""

    def evaluate(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        return [self._enrich(row) for row in super().evaluate(query, params)]

    @staticmethod
    def _enrich(row: dict[str, Any]) -> dict[str, Any]:
        rich = {"top_values": _top_values(), "sample_rows": _sample_rows()}
        out = dict(row)
        out["c"] = {**row["c"], **rich}
        if "t" in row:
            out["t"] = {**row["t"], "sample_rows": _sample_rows()}
        out["properties"] = [{**p, **rich} for p in row.get("properties") or []]
        return out


def test_source_evidence_omits_profiled_values_from_rich_graph() -> None:
    """Guard: expected green; evidence never carries top values or sample rows."""
    from sema.binding.candidates import enumerate_scope
    from sema.binding.evidence import build_source_column_evidence

    driver = RichEvidenceDriver(default_graph())
    params = {"catalog": CATALOG, "schemas": [SCHEMA], "table_pattern": None}
    rows = driver.evaluate(CypherQueries.enumerate_scope_columns(), params)
    assert rows and all(TOP_VALUE in str(row["c"]) for row in rows)
    assert all(SAMPLE_VALUE in str(row["t"]) for row in rows)

    scope = enumerate_scope(driver, DiscoveryScope(catalog=CATALOG, schemas=[SCHEMA]))
    assert len(scope) == 9

    for column in scope:
        dumped = build_source_column_evidence(column, scope).model_dump_json()
        assert TOP_VALUE not in dumped, column.column_ref
        assert SAMPLE_VALUE not in dumped, column.column_ref
        for key in VALUE_KEYS:
            assert key not in dumped, (column.column_ref, key)
