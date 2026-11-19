"""Done-when 4 (N6), in the graph: each coded column's label source.

The identity fixture is built through the DuckDB connector with its dictionary, model mocked,
embeddings skipped. A code list that is also a loaded relation labels from that relation's
display-role text column through a join; a dictionary-only list keeps its `Term` labels and
no edge; two non-constant text columns with no display role give no label.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
from neo4j import Driver, GraphDatabase

from tests.support.connector_fixture_identity import (
    HATCH_LIST_TEXTS,
    IDENTITY_SCHEMA,
    RIG_TITLES,
    column_values,
)
from tests.support.connector_fixture_identity_runs import IdentityBuild, build_identity_graph

pytestmark = pytest.mark.integration

VERIFIED = "verified"
_LABELED_BY = (
    "MATCH (c:Column {schema_name: $schema})-[l:LABELED_BY]->(lc:Column) "
    "RETURN c.table_name AS t, c.name AS c, lc.table_name AS lt, lc.name AS lc, "
    "properties(l) AS props"
)
_MEMBERS = (
    "MATCH (c:Column {schema_name: $schema})-[h:HAS_VALUE_SET]->(:ValueSet)<-[:MEMBER_OF]-"
    "(t:Term) WHERE h.verification = $verified "
    "RETURN c.table_name AS t, c.name AS c, t.code AS code, t.label AS label"
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
    return build_identity_graph(driver, tmp_path_factory.mktemp("identity_labels"))


def _rows(driver: Driver, query: str) -> list[dict[str, Any]]:
    with driver.session() as session:
        return [r.data() for r in session.run(query, schema=IDENTITY_SCHEMA, verified=VERIFIED)]


def labels_of(driver: Driver, table: str, column: str) -> list[tuple[tuple[str, str], Any]]:
    return [((r["lt"], r["lc"]), r["props"]) for r in _rows(driver, _LABELED_BY)
            if (r["t"], r["c"]) == (table, column)
            and r["props"].get("verification") == VERIFIED]


def member_labels(driver: Driver, table: str, column: str) -> dict[str, Any]:
    return {r["code"]: r["label"] for r in _rows(driver, _MEMBERS)
            if (r["t"], r["c"]) == (table, column)}


def present(table: str, column: str) -> set[str]:
    return {str(v) for v in column_values(table, column) if v is not None}


def test_a_loaded_code_list_labels_from_its_relations_display_column(
    built: IdentityBuild, driver: Driver,
) -> None:
    built.built()

    ((label, props),) = labels_of(driver, "hull_rev", "rig_type")

    assert label == ("rig_kind", "rig_title")
    assert props.get("via"), "the label names the join it is reached by"
    codes = present("hull_rev", "rig_type")
    held = member_labels(driver, "hull_rev", "rig_type")
    assert {code: held.get(code) for code in codes} == {c: RIG_TITLES[c] for c in codes}


def test_a_dictionary_only_list_keeps_its_labels_and_no_edge(built: IdentityBuild,
                                                             driver: Driver) -> None:
    built.built()

    assert labels_of(driver, "hull_cargo", "hatch_kind") == []
    codes = present("hull_cargo", "hatch_kind")
    held = member_labels(driver, "hull_cargo", "hatch_kind")
    assert {code: held.get(code) for code in codes} == {c: HATCH_LIST_TEXTS[c] for c in codes}


def test_two_non_constant_texts_and_no_display_role_give_no_label(
    built: IdentityBuild, driver: Driver,
) -> None:
    built.built()

    assert member_labels(driver, "berth", "dock_code"), "the binding is verified"
    assert labels_of(driver, "berth", "dock_code") == []
    assert labels_of(driver, "hull_rev", "rig_type"), "the positive case labels"
