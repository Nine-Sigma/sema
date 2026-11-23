"""Done-when 4 (N5), the graph half: inline value lists in the graph the build writes.

The declared-join fixture is built through the DuckDB connector with its dictionary and the
mocked model, embeddings skipped. A list the data holds links its column with its labels; a
list that misses an observed value links nothing; equal lists on two properties stay two value
sets; the list of an entity a declared join binds links only the bound relation's column; and
a stated domain of an entity in no join links every column it names, as before.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from neo4j import Driver, GraphDatabase

from tests.support.connector_fixture_declared import LID_LIST, TRAY_LIST, build_declared_graph
from tests.support.connector_fixture_identity import IDENTITY_SCHEMA
from tests.support.connector_fixture_identity_runs import IdentityBuild

pytestmark = pytest.mark.integration

VERIFIED = "verified"
LINKS = (
    "MATCH (c:Column {schema_name: $schema})-[h:HAS_VALUE_SET]->(vs:ValueSet) "
    "OPTIONAL MATCH (t:Term)-[:MEMBER_OF]->(vs) "
    "RETURN c.table_name AS t, c.name AS c, elementId(vs) AS vs, properties(h) AS props, "
    "collect([t.code, t.label]) AS members"
)
Link = tuple[str, dict[str, Any], dict[str, Any]]


@pytest.fixture(scope="module")
def driver() -> Iterator[Driver]:
    opened = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "graphrag")),
    )
    yield opened
    opened.close()


@dataclass
class Links:
    build: IdentityBuild
    by_column: dict[tuple[str, str], list[Link]] = field(default_factory=dict)

    def __contains__(self, column: tuple[str, str]) -> bool:
        return column in self.by_column


@pytest.fixture(scope="module")
def links(driver: Driver, tmp_path_factory: pytest.TempPathFactory) -> Links:
    found = Links(build_declared_graph(driver, tmp_path_factory.mktemp("inline_graph")))
    if found.build.error is not None:
        return found
    with driver.session() as session:
        for r in session.run(LINKS, schema=IDENTITY_SCHEMA):
            members = {str(code): label for code, label in r["members"] if code is not None}
            found.by_column.setdefault((r["t"], r["c"]), []).append(
                (r["vs"], dict(r["props"]), members))
    return found


def _verified(links: Links, table: str, column: str) -> list[Link]:
    links.build.built()
    return [link for link in links.by_column.get((table, column), [])
            if link[1].get("verification") == VERIFIED]


def test_a_list_the_data_holds_links_its_column_with_its_labels(
    links: Links,
) -> None:
    ((_, _, members),) = _verified(links, "crate_head", "lid_kind")

    assert members == LID_LIST


def test_a_list_that_misses_an_observed_value_links_nothing(
    links: Links,
) -> None:
    assert _verified(links, "crate_head", "lid_kind"), "the held list links"
    assert ("crate_head", "strap_kind") not in links


def test_equal_lists_on_two_properties_stay_two_value_sets(
    links: Links,
) -> None:
    ((lid, _, _),) = _verified(links, "crate_head", "lid_kind")
    ((wrap, _, _),) = _verified(links, "crate_head", "wrap_kind")

    assert lid != wrap


def test_a_bound_entitys_list_links_only_the_bound_relations_column(
    links: Links,
) -> None:
    assert _verified(links, "crate_head", "lid_kind"), "the bound relation's column links"
    assert ("crate_head_spare", "lid_kind") not in links


def test_an_unbound_entitys_stated_domain_links_every_column_it_names(
    links: Links,
) -> None:
    for table in ("tray_stock", "tray_return"):
        ((_, _, members),) = _verified(links, table, "tray_kind")
        assert members == TRAY_LIST
