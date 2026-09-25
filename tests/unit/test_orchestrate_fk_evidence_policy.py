"""FK detection under the evidence policy (design 5.3, 7.1 check 3).

``run_fk_detection`` issues value-bearing SQL (row counts, distinct counts,
distinct-value samples) once FK candidates exist. Under ``metadata_only`` the
recording connector must see none of it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.models.config import BuildConfig
from sema.pipeline.orchestrate_utils import run_fk_detection

pytestmark = pytest.mark.unit

CATALOG = "syn_cat"
SCHEMA = "clinical"
COLUMNS: dict[str, list[tuple[str, str]]] = {
    "patients": [("patients_id", "STRING"), ("diagnosis", "STRING"), ("age", "INT")],
    "samples": [("sample_id", "STRING"), ("patients_id", "STRING")],
}


class RecordingConnector:
    """Connector stand-in that records every SQL string it is asked to run."""

    def __init__(self) -> None:
        self.sql: list[str] = []

    def execute_query(self, query: str) -> list[tuple[Any, ...]]:
        self.sql.append(query)
        upper = query.upper()
        if "SELECT DISTINCT" in upper:
            return [("P1",), ("P2",)]
        if "COUNT(" in upper:
            return [(2,)]
        return []


def _loader(tables: list[str]) -> MagicMock:
    loader = MagicMock()
    loader._run_read.return_value = [
        {"name": name, "table_name": table, "catalog": CATALOG,
         "schema_name": SCHEMA, "data_type": data_type, "nullable": True,
         "comment": None}
        for table in tables for name, data_type in COLUMNS[table]
    ]
    return loader


def _run(policy: str) -> list[str]:
    connector = RecordingConnector()
    config = BuildConfig(
        catalog=CATALOG, schemas=[SCHEMA], evidence_policy=policy,
        enable_fk_detection=True,
    )
    run_fk_detection(
        _loader(["patients", "samples"]), connector, config, [SCHEMA], run_id="r-1",
    )
    return connector.sql


def test_full_policy_runs_fk_sampler_sql_for_candidate_tables() -> None:
    sql = _run("full")

    assert any("COUNT(*)" in q for q in sql)
    assert any("SELECT DISTINCT" in q for q in sql)


def test_metadata_only_issues_no_fk_sql() -> None:
    assert _run("metadata_only") == []
