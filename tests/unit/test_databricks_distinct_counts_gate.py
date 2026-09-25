"""``ProfilingConfig.enable_distinct_counts`` gates the per-column distinct scan.

Design 5.3: with the flag off, ``_build_profiling_assertions`` never calls
``_approx_distinct``, so no ``HAS_TOP_VALUES`` assertion is produced. The
``LIMIT 0`` top-k and sample queries are out of scope here.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import SecretStr

from sema.connectors.databricks import DatabricksConnector
from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.config import DatabricksConfig, ProfilingConfig
from tests.unit.test_databricks_sql_pin import CATALOG, RecordingConnection

pytestmark = pytest.mark.unit


def _extract(profiling: ProfilingConfig) -> tuple[list[str], list[Assertion]]:
    connection = RecordingConnection()
    databricks = DatabricksConfig(
        host="https://fake.example.invalid", http_path="/sql/fake",
        token=SecretStr("fake-token"),
    )
    with patch("sema.connectors.databricks.sql_connect", return_value=connection):
        connector = DatabricksConnector(config=databricks, profiling=profiling)
        assertions = connector.extract(CATALOG)
    return connection.sql, assertions


def _distinct_scans(sql: list[str]) -> list[str]:
    return [q for q in sql if "APPROX_COUNT_DISTINCT" in q.upper()]


def _top_values(assertions: list[Assertion]) -> list[Assertion]:
    return [a for a in assertions if a.predicate == AssertionPredicate.HAS_TOP_VALUES]


def test_distinct_counts_off_issues_no_distinct_scan_and_no_top_values() -> None:
    sql, assertions = _extract(ProfilingConfig(enable_distinct_counts=False))

    assert _distinct_scans(sql) == []
    assert _top_values(assertions) == []
    assert any("information_schema" in q for q in sql)


def test_distinct_counts_default_on_runs_scan_and_top_values() -> None:
    profiling = ProfilingConfig()
    assert profiling.enable_distinct_counts is True

    sql, assertions = _extract(profiling)

    assert len(_distinct_scans(sql)) == 4
    assert _top_values(assertions)
