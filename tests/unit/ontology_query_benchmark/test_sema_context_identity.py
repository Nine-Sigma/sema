"""Arm S: a declared entity identity is a relation the model selects from by name.

The context lists the entity with its columns and its key and holds no CTE body;
the S executor runs a statement that names the entity with the entity's rows.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import pytest

from showcase.ontology_query_benchmark.profile import Profile
from tests.unit.ontology_query_benchmark.runner_env import FakeModel
from tests.unit.ontology_query_benchmark.sema_bindings_env import EVENTS, PROMPT_FRAME_CHARS, RID
from tests.unit.ontology_query_benchmark.sema_context_env import (
    EVENTS_ENTITY,
    LIMITS,
    PEOPLE_ENTITY,
    QUESTION,
    _first_prompt,
    _overflows,
    _provider,
)
from tests.unit.ontology_query_benchmark.sema_context_identity_env import (
    EVENT_ENTITIES,
    EVENT_IDENTITY,
    _s_executor,
)

pytestmark = pytest.mark.unit


EVENT_COLUMNS = f"{RID}, site_S1, site_S1_label, first_day"


def _rows(runtime: Any, sql: str) -> list[tuple[Any, ...]]:
    execution = runtime.executor.execute(sql, 1000)
    assert execution.ok, execution.message
    return list(execution.rows)


def _ask(harness: SimpleNamespace, runtime: Any, sql: str) -> dict[str, Any]:
    question = harness.models.Question("e1", "regular", "How many events?", "scalar", None)
    model = FakeModel(harness, [f"```sql\n{sql}\n```"])
    return harness.workflow.run_question(question, runtime, model)


def test_the_context_lists_an_entity_as_a_table_with_its_key_and_no_cte_body(
    tmp_path: Path,
) -> None:
    ctx = _provider(tmp_path, [EVENTS_ENTITY], events_identity=EVENT_IDENTITY)("How many events?")
    keyed = [line for line in ctx.text.splitlines() if "event_key" in line]
    assert any(re.search(r"\bevent\b", line) for line in keyed), ctx.text
    assert "first_day" in ctx.text
    assert "GROUP BY" not in ctx.text.upper()
    assert "ROW_NUMBER" not in ctx.text.upper()


def test_a_statement_selecting_from_the_entity_name_runs_through_the_s_executor(
    tmp_path: Path, harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    provider = _provider(tmp_path, [EVENTS_ENTITY], events_identity=EVENT_IDENTITY)
    runtime, _ = _s_executor(harness, frozen, profile, provider)
    record = _ask(harness, runtime, "SELECT count(*) AS n FROM event")
    assert record["answer"] == [[len(EVENT_ENTITIES)]]


def test_a_statement_with_its_own_with_clause_still_reads_the_entity(
    tmp_path: Path, harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    provider = _provider(tmp_path, [EVENTS_ENTITY], events_identity=EVENT_IDENTITY)
    runtime, _ = _s_executor(harness, frozen, profile, provider)
    sql = ("WITH sited AS (SELECT * FROM event WHERE site_S1 IS NOT NULL) "
           "SELECT count(*) AS n FROM sited")  # fmt: skip
    assert _rows(runtime, sql) == [(3,)]


def test_the_entity_key_counts_each_entity_once_through_the_s_executor(
    tmp_path: Path, harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    provider = _provider(tmp_path, [EVENTS_ENTITY], events_identity=EVENT_IDENTITY)
    runtime, _ = _s_executor(harness, frozen, profile, provider)
    assert _rows(runtime, "SELECT count(DISTINCT event_key) AS n FROM event") == [(4,)]


def test_a_database_error_through_the_s_executor_is_a_repairable_sql_error(
    tmp_path: Path, harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    provider = _provider(tmp_path, [EVENTS_ENTITY], events_identity=EVENT_IDENTITY)
    runtime, _ = _s_executor(harness, frozen, profile, provider)
    for sql in ("SELECT no_such_column FROM event", "SELECT FROM WHERE ("):
        execution = runtime.executor.execute(sql, 1000)
        assert (execution.ok, execution.error_class) == (False, "sql_error")


def test_a_declared_entity_resolves_to_one_row_per_identity(
    tmp_path: Path, harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    provider = _provider(tmp_path, [EVENTS_ENTITY], events_identity=EVENT_IDENTITY)
    runtime, _ = _s_executor(harness, frozen, profile, provider)
    sql = f"SELECT {EVENT_COLUMNS} FROM event ORDER BY ALL NULLS LAST"
    assert _rows(runtime, sql) == EVENT_ENTITIES


def test_an_overlay_identity_resolves_the_same_entity_as_the_spec(
    tmp_path: Path, harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    overlay = {"entity_id": "event", "identity": EVENT_IDENTITY}
    supplied = _provider(tmp_path, [EVENTS_ENTITY], events_overlay=overlay)
    runtime, _ = _s_executor(harness, frozen, profile, supplied)
    sql = f"SELECT {EVENT_COLUMNS} FROM event ORDER BY ALL NULLS LAST"
    assert _rows(runtime, sql) == EVENT_ENTITIES


def test_a_fixture_entity_excludes_named_null_keys_and_ranks_preferred_values(
    tmp_path: Path, harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    identity = {
        "key_columns": [RID, "site_S1"], "exclude_null_keys": [RID],
        "attributes": [{"column": "site_S1_label", "rule": "min_by", "by": "event_day",
                        "prefer": {"column": "site_S1_mapping_type",
                                   "value": ["other", "preferred"]}}],
    }  # fmt: skip
    provider = _provider(tmp_path, [EVENTS_ENTITY], events_identity=identity)
    runtime, _ = _s_executor(harness, frozen, profile, provider)
    sql = f"SELECT {RID}, site_S1, site_S1_label FROM event ORDER BY ALL NULLS LAST"
    assert _rows(runtime, sql) == [
        ("r1", "P1", "North a"), ("r1", "P2", "South x"), ("r1", None, None),
    ]


UNSAFE_IDENTITY = {
    "key_columns": [RID],
    "attributes": [{"column": "event_day", "rule": "min", "name": "first day"}],
}


@pytest.mark.parametrize(("dialect", "quoted"), [
    ("databricks", "`first day`"), ("spark", "`first day`"),
    ("duckdb", '"first day"'), ("postgres", '"first day"'),
])  # fmt: skip
def test_the_warehouse_receives_unsafe_identifiers_in_its_own_quote(
    tmp_path: Path, harness: SimpleNamespace, frozen: Path, profile: Profile,
    dialect: str, quoted: str,
) -> None:  # fmt: skip
    provider = _provider(tmp_path, [EVENTS_ENTITY], events_identity=UNSAFE_IDENTITY,
                         sql_dialect=dialect)  # fmt: skip
    runtime, warehouse = _s_executor(harness, frozen, profile, provider)
    runtime.executor.execute(f"SELECT {RID} FROM event", 1000)
    (sent,) = warehouse.sqls
    assert f"AS {quoted}" in sent
    assert ('"' if quoted.startswith("`") else "`") not in sent


def test_a_duckdb_rendered_unsafe_identifier_executes_in_duckdb(
    tmp_path: Path, harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    provider = _provider(tmp_path, [EVENTS_ENTITY], events_identity=UNSAFE_IDENTITY,
                         sql_dialect="duckdb")  # fmt: skip
    runtime, _ = _s_executor(harness, frozen, profile, provider)
    sql = f'SELECT {RID}, "first day" FROM event ORDER BY ALL NULLS LAST'
    assert _rows(runtime, sql) == [("r1", 1), (None, 4)]


def test_an_entity_reading_a_denied_column_is_not_executable(
    tmp_path: Path, harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    provider = _provider(tmp_path, [EVENTS_ENTITY], events_identity=EVENT_IDENTITY,
                         denied={EVENTS: ("site_S1_mapping_type",)})  # fmt: skip
    runtime, _ = _s_executor(harness, frozen, profile, provider)
    execution = runtime.executor.execute("SELECT count(*) AS n FROM event", 1000)
    assert (execution.ok, execution.error_class) == (False, "sql_error")


def test_an_entity_identity_outside_the_retrieved_tables_is_not_listed(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [PEOPLE_ENTITY], events_identity=EVENT_IDENTITY)(QUESTION)
    assert "event_key" not in ctx.text


def test_an_entity_listing_counts_toward_the_context_budget(tmp_path: Path) -> None:
    plain = len(_provider(tmp_path / "a", [EVENTS_ENTITY])("q").text)
    needed = (plain + PROMPT_FRAME_CHARS + 1) / LIMITS.conservative_chars_per_token
    limits = _first_prompt(math.ceil(needed) + 1)
    fits = _provider(tmp_path / "b", [EVENTS_ENTITY], limits=limits)("q")
    grown = _provider(tmp_path / "c", [EVENTS_ENTITY], limits=limits,
                      events_identity=EVENT_IDENTITY)("q")  # fmt: skip
    assert not _overflows(fits.findings)
    assert "event_key" in grown.text
    assert _overflows(grown.findings)


@pytest.mark.parametrize(("denied", "column"), [
    ("site_S1_mapping_type", "site_S1_mapping_type"), ("EVENT_DAY", "event_day"),
])  # fmt: skip
def test_an_entity_identity_reading_a_denied_column_is_withheld_with_a_finding(
    tmp_path: Path, denied: str, column: str
) -> None:
    provider = _provider(tmp_path, [EVENTS_ENTITY], events_identity=EVENT_IDENTITY)
    ctx = provider("How many events?", lambda name: name.lower() == denied.lower())
    assert "event_key" not in ctx.text
    assert f"  cat.sch.{EVENTS}: " in ctx.text
    withheld = [(f.relation, f.column) for f in ctx.findings if "entity identity" in f.message]
    assert withheld == [(f"cat.sch.{EVENTS}", column)]
