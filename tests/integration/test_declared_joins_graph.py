"""Done-when 3 (N4), the graph half: the declared-join fixture built into Neo4j.

Built through the DuckDB connector with the fixture's dictionary and the mocked model,
embeddings skipped. The graph holds one `JoinPath` per accepted join, with one predicate per
column pair, and none for a rejected or open join; the core context the query path renders
prints the accepted join with both pairs joined by `AND`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import pytest
from neo4j import Driver, GraphDatabase

from sema.consumers.nl2sql.prompting import render_sections
from sema.pipeline.context import prune_to_sco
from sema.pipeline.retrieval import RetrievalEngine
from tests.support.connector_fixture_declared import (
    ACCEPTED,
    DICTIONARY_ENTRIES,
    OPEN,
    REJECTED,
    build_declared_graph,
)
from tests.support.connector_fixture_identity_runs import IdentityBuild

pytestmark = pytest.mark.integration

SOURCE = "dictionary_declared"
QUESTION = "How many crate line rows does each crate head hold?"
JOIN_PATHS = "MATCH (jp:JoinPath) RETURN jp.join_predicates AS predicates, jp.source AS source"
JoinRow = tuple[list[dict[str, Any]], Any]


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
    return build_declared_graph(driver, tmp_path_factory.mktemp("declared_graph"))


def _join_paths(driver: Driver) -> list[JoinRow]:
    with driver.session() as session:
        rows = [record.data() for record in session.run(JOIN_PATHS)]
    return [(json.loads(r["predicates"]) if isinstance(r["predicates"], str) else r["predicates"],
             r["source"]) for r in rows]


def _key(predicates: list[dict[str, Any]]) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """The referencing table, the referenced table and the (local, remote) pairs."""
    return (predicates[0]["right_table"], predicates[0]["left_table"],
            tuple((p["right_column"], p["left_column"]) for p in predicates))


def test_one_join_path_per_accepted_join_with_one_predicate_per_pair(
    built: IdentityBuild, driver: Driver,
) -> None:
    built.built()
    composite = [(p, source) for p, source in _join_paths(driver) if len(p) > 1]

    expected = sorted((e.relation, ACCEPTED[e.relation], e.pairs)
                      for e in DICTIONARY_ENTRIES if e.relation in ACCEPTED)
    assert sorted(_key(p) for p, _ in composite) == expected
    assert {source for _, source in composite} == {SOURCE}


def test_no_join_path_for_a_rejected_or_open_join(built: IdentityBuild, driver: Driver) -> None:
    built.built()
    paths = _join_paths(driver)

    assert paths, "the build wrote no join path"
    for predicates, source in paths:
        if source == SOURCE or len(predicates) > 1:
            assert _key(predicates)[0] not in {*REJECTED, *OPEN}, predicates


def test_the_core_context_prints_the_join_with_both_pairs_joined_by_and(
    built: IdentityBuild, driver: Driver,
) -> None:
    built.built()
    sco = prune_to_sco(RetrievalEngine(driver=driver).retrieve(QUESTION))

    lines = [line.strip() for part in render_sections(sco, "duckdb") for line in part.splitlines()]

    expected = ("crate_head.crate_no = crate_line.crate_no AND "
                "crate_head.seal_day = crate_line.seal_day")
    assert any(expected in line for line in lines), lines
