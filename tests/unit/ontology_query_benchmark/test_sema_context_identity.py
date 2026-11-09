"""Arm S context: a declared entity identity renders as an executable CTE."""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pytest

from tests.unit.ontology_query_benchmark.test_sema_bindings import EVENTS, PROMPT_FRAME_CHARS, RID
from tests.unit.ontology_query_benchmark.test_sema_context import (
    EVENTS_ENTITY,
    LIMITS,
    PEOPLE_ENTITY,
    QUESTION,
    _first_prompt,
    _overflows,
    _provider,
)

pytestmark = pytest.mark.unit


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


def _entity_cte(text: str, entity: str) -> str:
    lines = text.splitlines()
    (at,) = [i for i, line in enumerate(lines) if line.startswith(f"  ENTITY {entity} (executable")]
    return lines[at + 1].strip()


def _event_entities(cte: str) -> list[tuple[object, ...]]:
    con = duckdb.connect()
    con.execute("ATTACH ':memory:' AS cat")
    con.execute("CREATE SCHEMA cat.sch")
    con.execute(f"CREATE TABLE cat.sch.{EVENTS} (row_id VARCHAR, {RID} VARCHAR, site_S1 VARCHAR, "
                "site_S1_label VARCHAR, site_S1_mapping_type VARCHAR, event_day INTEGER)")
    con.executemany(f"INSERT INTO cat.sch.{EVENTS} VALUES (?, ?, ?, ?, ?, ?)", EVENT_ROWS)
    return con.execute(f"{cte} SELECT * FROM event ORDER BY ALL NULLS LAST").fetchall()


def test_a_declared_entity_identity_is_rendered_as_an_executable_cte(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [EVENTS_ENTITY], events_identity=EVENT_IDENTITY)("How many events?")
    assert f"  ENTITY event (executable; one row per {RID}, site_S1)" in ctx.text
    assert _event_entities(_entity_cte(ctx.text, "event")) == [
        ("r1", "P1", "North a", 2), ("r1", "P2", "South y", 1),
        ("r1", None, None, 3), (None, "P1", "North p", 4),
    ]


def test_an_overlay_identity_renders_the_same_executable_entity_as_the_spec(
    tmp_path: Path,
) -> None:
    overlay = {"entity_id": "event", "identity": EVENT_IDENTITY}
    spec = _provider(tmp_path / "spec", [EVENTS_ENTITY], events_identity=EVENT_IDENTITY)("q")
    supplied = _provider(tmp_path / "overlay", [EVENTS_ENTITY], events_overlay=overlay)("q")
    assert _entity_cte(supplied.text, "event") == _entity_cte(spec.text, "event")
    assert _event_entities(_entity_cte(supplied.text, "event")) == [
        ("r1", "P1", "North a", 2), ("r1", "P2", "South y", 1),
        ("r1", None, None, 3), (None, "P1", "North p", 4),
    ]


def test_a_fixture_entity_excludes_named_null_keys_and_ranks_preferred_values(
    tmp_path: Path,
) -> None:
    identity = {
        "key_columns": [RID, "site_S1"], "exclude_null_keys": [RID],
        "attributes": [{"column": "site_S1_label", "rule": "min_by", "by": "event_day",
                        "prefer": {"column": "site_S1_mapping_type",
                                   "value": ["other", "preferred"]}}],
    }  # fmt: skip
    ctx = _provider(tmp_path, [EVENTS_ENTITY], events_identity=identity)("q")
    assert _event_entities(_entity_cte(ctx.text, "event")) == [
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
def test_an_unsafe_identifier_renders_with_the_warehouse_identifier_quote(
    tmp_path: Path, dialect: str, quoted: str
) -> None:
    ctx = _provider(tmp_path, [EVENTS_ENTITY], events_identity=UNSAFE_IDENTITY,
                    sql_dialect=dialect)("q")  # fmt: skip
    cte = _entity_cte(ctx.text, "event")
    assert f"AS {quoted}" in cte
    assert ('"' if quoted.startswith("`") else "`") not in cte


def test_a_duckdb_rendered_unsafe_identifier_executes_in_duckdb(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [EVENTS_ENTITY], events_identity=UNSAFE_IDENTITY,
                    sql_dialect="duckdb")("q")  # fmt: skip
    assert _event_entities(_entity_cte(ctx.text, "event")) == [("r1", 1), (None, 4)]


def test_an_entity_identity_outside_the_retrieved_tables_is_not_rendered(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [PEOPLE_ENTITY], events_identity=EVENT_IDENTITY)(QUESTION)
    assert "ENTITY event" not in ctx.text


def test_an_entity_cte_counts_toward_the_context_budget(tmp_path: Path) -> None:
    plain = len(_provider(tmp_path / "a", [EVENTS_ENTITY])("q").text)
    needed = (plain + PROMPT_FRAME_CHARS + 1) / LIMITS.conservative_chars_per_token
    limits = _first_prompt(math.ceil(needed) + 1)
    fits = _provider(tmp_path / "b", [EVENTS_ENTITY], limits=limits)("q")
    grown = _provider(tmp_path / "c", [EVENTS_ENTITY], limits=limits,
                      events_identity=EVENT_IDENTITY)("q")  # fmt: skip
    assert not _overflows(fits.findings)
    assert "ENTITY event (executable" in grown.text
    assert _overflows(grown.findings)



@pytest.mark.parametrize(("denied", "column"), [
    ("site_S1_mapping_type", "site_S1_mapping_type"), ("EVENT_DAY", "event_day"),
])  # fmt: skip
def test_an_entity_identity_reading_a_denied_column_is_withheld_with_a_finding(
    tmp_path: Path, denied: str, column: str
) -> None:
    provider = _provider(tmp_path, [EVENTS_ENTITY], events_identity=EVENT_IDENTITY)
    ctx = provider("How many events?", lambda name: name.lower() == denied.lower())
    assert "ENTITY event" not in ctx.text
    assert f"  cat.sch.{EVENTS}: " in ctx.text
    withheld = [(f.relation, f.column) for f in ctx.findings if "entity identity" in f.message]
    assert withheld == [(f"cat.sch.{EVENTS}", column)]
