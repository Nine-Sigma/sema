"""Done-when 8 (C1): the identity fixture end to end, scored and rendered from its graph.

The fixture is built through the DuckDB connector with its dictionary and the mocked model,
embeddings skipped. `sema eval discovery` scores the graph against the fixture's claims for
the six slice C kinds. Then the path arm `S` takes, `render_sections` on the `prune_to_sco`
result for a neutral question, shows a key line, an entity line and a label; every number an
entity line states, and its claim that the ordering column picks one row of each entity, is
checked by a count this test runs on the fixture through DuckDB.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest
from click.testing import CliRunner
from neo4j import Driver, GraphDatabase

from sema.consumers.nl2sql.prompting import render_sections
from sema.pipeline.context import prune_to_sco
from sema.pipeline.retrieval import RetrievalEngine
from tests.support.connector_fixture_identity import IDENTITY_CATALOG, IDENTITY_SCHEMA
from tests.support.connector_fixture_identity_runs import (
    IdentityBuild,
    build_identity_graph,
    identity_claims,
)
from tests.support.connector_fixture_scorecard import card_rows

pytestmark = pytest.mark.integration

CLAIMS = identity_claims()
QUESTION = "Which crane lifts the most tons, and what rig does each hull carry?"
ENTITY_LINE = re.compile(
    r"^(?P<table>\S+): several rows can hold one (?P<entity>.+?); one (?P=entity) per distinct "
    r"(?P<key>[\w, ]+?)(?:; (?P<nulls>\d+) rows have a NULL in (?P<null_key>[\w, ]+?))?"
    r"(?:; each (?P=entity) has exactly one row per (?P<by>\w+), so the greatest (?P=by) "
    r"picks one row of each)?$"
)


@pytest.fixture(scope="module")
def driver() -> Iterator[Driver]:
    opened = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "graphrag")),
    )
    yield opened
    opened.close()


@pytest.fixture(scope="module")
def built(driver: Driver, tmp_path_factory: pytest.TempPathFactory) -> IdentityBuild:
    return build_identity_graph(driver, tmp_path_factory.mktemp("identity_end_to_end"))


def test_the_six_slice_c_kinds_score_equal_from_the_graph(built: IdentityBuild,
                                                          tmp_path: Path) -> None:
    from sema.cli_eval import eval_group

    built.built()
    oracle, out = tmp_path / "oracle.jsonl", tmp_path / "card.md"
    oracle.write_text("".join(
        json.dumps({"kind": kind, "key": list(key), "value": value, "basis": "data",
                    "cite": "fixture"}) + "\n"
        for kind, cells in CLAIMS.items() for key, value in cells.items()
    ))

    result = CliRunner().invoke(eval_group, [
        "discovery", "--config", str(built.config_path), "--oracle-spec", str(oracle),
        "--out", str(out),
    ])

    assert result.exit_code == 0, result.output
    rows = card_rows(out.read_text())
    for kind, claims in CLAIMS.items():
        assert [rows[kind][cell] for cell in ("equal", "missing", "wrong")] == [
            str(len(claims)), "0", "0"], kind


def context_lines(driver: Driver) -> list[str]:
    sco = prune_to_sco(RetrievalEngine(driver=driver).retrieve(QUESTION))
    return [line.strip() for part in render_sections(sco, "duckdb") for line in part.splitlines()]


def _count(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    (row,) = con.execute(sql).fetchall()
    return int(row[0])


def _columns(text: str) -> list[str]:
    return [c.strip() for c in text.split(",")]


def test_the_context_shows_a_key_an_entity_and_a_label(built: IdentityBuild,
                                                       driver: Driver) -> None:
    built.built()

    lines = context_lines(driver)

    assert "KEYS AND ENTITIES:" in lines
    assert any(": one row per " in line for line in lines), lines
    assert any(ENTITY_LINE.match(line) for line in lines), lines
    assert any("label: " in line for line in lines), lines


def test_each_number_an_entity_line_states_equals_a_count_on_the_fixture(
    built: IdentityBuild, driver: Driver,
) -> None:
    built.built()
    stated = [m for line in context_lines(driver) if (m := ENTITY_LINE.match(line))]
    assert any(m["nulls"] for m in stated), "no entity line states a NULL-key count"
    assert any(m["by"] for m in stated), "no entity line states an ordering column"

    with duckdb.connect(str(built.database), read_only=True) as con:
        for m in stated:
            table = m["table"]
            assert table.startswith(f"{IDENTITY_CATALOG}.{IDENTITY_SCHEMA}."), table
            key = _columns(m["key"])
            rows = _count(con, f"SELECT COUNT(*) FROM {table}")
            groups = _count(con, f"SELECT COUNT(*) FROM (SELECT DISTINCT {', '.join(key)} "
                                 f"FROM {table})")
            assert groups < rows, f"{table}: several rows hold one entity"
            if m["nulls"]:
                nulls = " OR ".join(f"{c} IS NULL" for c in _columns(m["null_key"]))
                assert int(m["nulls"]) == _count(
                    con, f"SELECT COUNT(*) FROM {table} WHERE {nulls}"), m.group(0)
            if m["by"]:
                ordered = ", ".join([*key, m["by"]])
                assert _count(con, f"SELECT COUNT(*) FROM (SELECT DISTINCT {ordered} "
                                   f"FROM {table})") == rows, m.group(0)
