"""Shared setup for test_sema_context_identity.py and the tests that reuse it."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb

from showcase.ontology_query_benchmark import runner_utils
from showcase.ontology_query_benchmark.profile import Profile
from showcase.ontology_query_benchmark.sema_context import SContextProvider
from tests.unit.ontology_query_benchmark.sema_bindings_env import EVENTS, RID

EVENT_IDENTITY = {
    "key_columns": [RID, "site_S1"],
    "attributes": [
        {"column": "site_S1_label", "rule": "min_by", "by": "event_day",
         "prefer": {"column": "site_S1_mapping_type", "value": "preferred"}},
        {"column": "event_day", "rule": "min", "name": "first_day"},
    ],
}  # fmt: skip
EVENT_ROWS = [
    ("e1", "r1", "P1", "North b", "other", 2),
    ("e2", "r1", "P1", "North a", "other", 2),
    ("e3", "r1", "P2", "South x", "other", 1),
    ("e4", "r1", "P2", "South y", "preferred", 5),
    ("e5", "r1", None, None, None, 3),
    ("e6", None, "P1", "North p", "other", 4),
    ("e7", None, "P1", "North q", "other", 6),
]
EVENT_ENTITIES = [
    ("r1", "P1", "North a", 2), ("r1", "P2", "South y", 1),
    ("r1", None, None, 3), (None, "P1", "North p", 4),
]  # fmt: skip


class DuckExecutor:
    """The warehouse executor contract: every database failure is a classified result."""

    def __init__(self, harness: SimpleNamespace) -> None:
        self.harness, self.sqls = harness, []
        self.con = duckdb.connect()
        self.con.execute("ATTACH ':memory:' AS cat")
        self.con.execute("CREATE SCHEMA cat.sch")
        self.con.execute(
            f"CREATE TABLE cat.sch.{EVENTS} (row_id VARCHAR, {RID} VARCHAR, site_S1 VARCHAR, "
            "site_S1_label VARCHAR, site_S1_mapping_type VARCHAR, event_day INTEGER)"
        )
        self.con.executemany(f"INSERT INTO cat.sch.{EVENTS} VALUES (?, ?, ?, ?, ?, ?)", EVENT_ROWS)

    def execute(self, sql: str, row_limit: int) -> Any:
        self.sqls.append(sql)
        models = self.harness.models
        try:
            cursor = self.con.execute(sql)
            rows = tuple(tuple(r) for r in cursor.fetchmany(row_limit))
        except duckdb.Error as exc:
            return models.Execution(False, (), (), 0.0, "sql_error", str(exc)[:200])
        return models.Execution(True, tuple(d[0] for d in cursor.description), rows, 0.0)


def _s_executor(
    harness: SimpleNamespace, frozen: Path, profile: Profile, provider: SContextProvider
) -> tuple[Any, DuckExecutor]:
    """The executor arm S hands the harness, over an in-memory warehouse."""
    warehouse = DuckExecutor(harness)
    runtimes = runner_utils.arm_runtimes(
        harness, frozen, profile.harness, {"S": warehouse}, provider, profile.limits
    )
    return runtimes["S"], warehouse
