"""Arm S: the entity key is counted, never returned, when it stands in for an identifier.

``<entity>_key`` hashes the key parts, so a statement returning it returns a reversible
stand-in for every key part. When the run's policy treats a key part as an identifier,
the S executor refuses such a statement as a repairable error. An entity whose key name
equals one of its own projected columns is never exposed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from showcase.ontology_query_benchmark.profile import Profile
from showcase.ontology_query_benchmark.sema_entity_utils import EntityExecutor
from tests.unit.ontology_query_benchmark.sema_bindings_env import EVENTS, RID
from tests.unit.ontology_query_benchmark.sema_context_env import EVENTS_ENTITY, _provider
from tests.unit.ontology_query_benchmark.sema_context_identity_env import (
    EVENT_ENTITIES,
    EVENT_IDENTITY,
    _s_executor,
)
from tests.unit.ontology_query_benchmark.sema_entity_lint_env import ENTITIES, DuckWarehouse

pytestmark = pytest.mark.unit

VISIT_ENTITIES = 4
IDENTIFIERS = frozenset({"visit_id", "document id"})


def _is_identifier(name: str) -> bool:
    return name.lower() in IDENTIFIERS


def _executor(
    harness: SimpleNamespace, is_identifier: Any = _is_identifier
) -> tuple[EntityExecutor, DuckWarehouse]:
    warehouse = DuckWarehouse(harness)
    executor = EntityExecutor(harness, warehouse, ENTITIES, "duckdb", is_identifier=is_identifier)
    return executor, warehouse


@pytest.mark.parametrize(("sql", "key"), [
    ("SELECT visit_key FROM visit", "visit_key"),
    ("SELECT v.visit_key AS k, COUNT(*) AS n FROM visit v GROUP BY 1", "visit_key"),
    ("WITH x AS (SELECT visit_key FROM visit) SELECT visit_key FROM x", "visit_key"),
    ("SELECT site FROM visit UNION ALL SELECT visit_key FROM visit", "visit_key"),
    ("SELECT document_key FROM document", "document_key"),
])  # fmt: skip
def test_returning_an_identifier_bearing_key_is_a_repairable_error(
    harness: SimpleNamespace, sql: str, key: str
) -> None:
    executor, warehouse = _executor(harness)
    execution = executor.execute(sql, 1000)
    assert (execution.ok, execution.error_class) == (False, "sql_error")
    assert key in execution.message
    assert "counted, not returned" in execution.message
    assert warehouse.sqls == []


@pytest.mark.parametrize(("sql", "expected"), [
    ("SELECT COUNT(DISTINCT visit_key) AS n FROM visit", [(VISIT_ENTITIES,)]),
    ("SELECT COUNT(*) AS n FROM visit GROUP BY visit_key HAVING COUNT(*) > 1", []),
    ("SELECT COUNT(*) AS n FROM visit a JOIN visit b ON a.visit_key = b.visit_key",
     [(VISIT_ENTITIES,)]),
    ("SELECT COUNT(*) AS n FROM visit WHERE visit_key IS NOT NULL", [(VISIT_ENTITIES,)]),
    ("WITH k AS (SELECT visit_key FROM visit) SELECT COUNT(*) AS n FROM k",
     [(VISIT_ENTITIES,)]),
    ("SELECT MAX(n) AS m FROM (SELECT visit_key, COUNT(*) AS n FROM visit GROUP BY visit_key)",
     [(1,)]),
])  # fmt: skip
def test_using_an_identifier_bearing_key_inside_the_statement_runs(
    harness: SimpleNamespace, sql: str, expected: list[tuple[Any, ...]]
) -> None:
    executor, _ = _executor(harness)
    execution = executor.execute(sql, 1000)
    assert execution.ok, execution.message
    assert list(execution.rows) == expected


def test_a_key_over_no_identifier_may_be_returned(harness: SimpleNamespace) -> None:
    executor, _ = _executor(harness, is_identifier=lambda _name: False)
    execution = executor.execute("SELECT visit_key FROM visit", 1000)
    assert execution.ok, execution.message
    assert len(execution.rows) == VISIT_ENTITIES


@dataclass(frozen=True)
class IdentifierPolicy:
    """A run policy that marks some names as identifiers, as the frozen harness's does."""

    identifiers: tuple[str, ...]

    def is_denied(self, name: str) -> bool:
        return False

    def is_identifier(self, name: str) -> bool:
        return name.lower() in {i.lower() for i in self.identifiers}


def test_the_run_policy_identifiers_reach_the_s_executor(
    tmp_path: Path, harness: SimpleNamespace, frozen: Path, profile: Profile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    load = harness.config.load_bench_config
    monkeypatch.setattr(harness.config, "load_bench_config", lambda path: dataclasses.replace(
        load(path), policy=IdentifierPolicy((RID.upper(),))))  # fmt: skip
    provider = _provider(tmp_path, [EVENTS_ENTITY], events_identity=EVENT_IDENTITY)
    runtime, _ = _s_executor(harness, frozen, profile, provider)
    returned = runtime.executor.execute("SELECT event_key FROM event", 1000)
    counted = runtime.executor.execute("SELECT COUNT(DISTINCT event_key) AS n FROM event", 1000)
    assert (returned.ok, returned.error_class) == (False, "sql_error")
    assert "counted, not returned" in returned.message
    assert counted.ok and list(counted.rows) == [(len(EVENT_ENTITIES),)]


CLASHING_IDENTITY = {
    "key_columns": [RID, "site_S1"],
    "attributes": [{"column": "event_day", "rule": "min", "name": "EVENT_KEY"}],
}


def test_an_entity_whose_key_clashes_is_withheld_from_the_context_with_a_finding(
    tmp_path: Path,
) -> None:
    ctx = _provider(tmp_path, [EVENTS_ENTITY], events_identity=CLASHING_IDENTITY)("q")
    assert "event_key" not in ctx.text.lower()
    clashes = [f for f in ctx.findings if "entity identity" in f.message]
    assert [(f.relation, f.column) for f in clashes] == [(f"cat.sch.{EVENTS}", "event_key")]
    assert "EVENT_KEY" in clashes[0].message


def test_an_entity_whose_key_clashes_is_not_exposed_to_the_s_executor(
    tmp_path: Path, harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    provider = _provider(tmp_path, [EVENTS_ENTITY], events_identity=CLASHING_IDENTITY)
    runtime, warehouse = _s_executor(harness, frozen, profile, provider)
    execution = runtime.executor.execute("SELECT COUNT(DISTINCT event_key) AS n FROM event", 1000)
    assert (execution.ok, execution.error_class) == (False, "sql_error")
    assert "WITH" not in warehouse.sqls[-1].upper()
