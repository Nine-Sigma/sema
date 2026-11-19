"""Done-when 2 (N4), in the graph: each relation's key on its `Table` node.

The identity fixture is built through the DuckDB connector with its dictionary, model mocked
and embeddings skipped. Each relation with a key holds `Table.key_columns` and
`Table.key_verification`, and one accepted `has_key` assertion per key; a relation with no
key, or with only an open one, holds neither property.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import pytest
from neo4j import Driver, GraphDatabase

from sema.models.physical_key import CanonicalRef
from tests.support.connector_fixture_identity import IDENTITY_SCHEMA, TABLES
from tests.support.connector_fixture_identity_runs import (
    KEYED,
    NULL_PART_KEYS,
    IdentityBuild,
    build_identity_graph,
)

pytestmark = pytest.mark.integration

ACCEPTED = ("proven", "supported")
SUPPORTED_ONLY = {"slip_b"}
_TABLES = ("MATCH (t:Table {schema_name: $schema}) "
           "RETURN t.name AS t, t.key_columns AS key, t.key_verification AS grade")
_KEYS = ("MATCH (a:Assertion {predicate: 'has_key'}) "
         "RETURN a.subject_ref AS subject, a.payload AS payload")


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
    return build_identity_graph(driver, tmp_path_factory.mktemp("identity_keys"))


def _rows(driver: Driver, query: str) -> list[dict[str, Any]]:
    with driver.session() as session:
        return [record.data() for record in session.run(query, schema=IDENTITY_SCHEMA)]


def table_keys(driver: Driver) -> dict[str, tuple[list[str] | None, str | None]]:
    return {r["t"]: (r["key"], r["grade"]) for r in _rows(driver, _TABLES)}


def accepted_assertions(driver: Driver) -> dict[str, list[list[str]]]:
    found: dict[str, list[list[str]]] = {}
    for r in _rows(driver, _KEYS):
        payload = json.loads(r["payload"]) if isinstance(r["payload"], str) else r["payload"]
        if payload.get("verification") in ACCEPTED:
            found.setdefault(CanonicalRef.parse(r["subject"]).table, []).append(
                sorted(payload["columns"]))
    return found


def test_each_relation_with_a_key_holds_it_on_its_table(built: IdentityBuild,
                                                         driver: Driver) -> None:
    built.built()
    keys = table_keys(driver)

    assert set(keys) == set(TABLES)
    for table, columns in KEYED.items():
        grade = "supported" if table in SUPPORTED_ONLY else "proven"
        held, verification = keys[table]
        assert (sorted(held or []), verification) == (sorted(columns), grade), table


def test_a_relation_with_no_key_or_an_open_one_holds_none(built: IdentityBuild,
                                                          driver: Driver) -> None:
    built.built()
    keys = table_keys(driver)

    for table in (*NULL_PART_KEYS, "gate_tally"):
        assert keys[table] == (None, None), table
    assert keys["berth"][0] == ["berth_no"]


def test_one_has_key_assertion_per_key(built: IdentityBuild, driver: Driver) -> None:
    built.built()

    assertions = accepted_assertions(driver)

    assert assertions == {table: [sorted(columns)] for table, columns in KEYED.items()}
