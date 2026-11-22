"""Done-when 3 (N5), in the graph: entity edges and `has_identity` assertions.

The identity fixture is built through the DuckDB connector with its dictionary, the model
mocked (`IdentityFakeClient`: L2 names one entity `Hull` on `hull_rev` and `hull_cargo`, and
each proposal call gets the fixture's proposal), embeddings skipped. The step writes the F6
properties on `ENTITY_ON_TABLE` edges, and an edge L2 wrote before it ran never carries
`owner = false`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import pytest
from neo4j import Driver, GraphDatabase

from sema.models.physical_key import CanonicalRef
from sema.models.target.identity import EntityIdentity
from tests.support.connector_fixture_identity import (
    IDENTITY_SCHEMA,
    L2_ENTITIES,
    PROPOSALS,
    null_row_count,
)
from tests.support.connector_fixture_identity_runs import (
    COARSER,
    IdentityBuild,
    build_identity_graph,
    ordered_by,
)

pytestmark = pytest.mark.integration

_EDGES = ("MATCH (e:Entity)-[r:ENTITY_ON_TABLE]->(t:Table {schema_name: $schema}) "
          "RETURN e.name AS entity, t.name AS t, properties(r) AS props")
_IDENTITIES = ("MATCH (a:Assertion {predicate: 'has_identity'}) "
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
    return build_identity_graph(driver, tmp_path_factory.mktemp("identity_entities"))


def _rows(driver: Driver, query: str) -> list[dict[str, Any]]:
    with driver.session() as session:
        return [record.data() for record in session.run(query, schema=IDENTITY_SCHEMA)]


def edges(driver: Driver) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["entity"], r["t"]): r["props"] for r in _rows(driver, _EDGES)}


def edge(driver: Driver, table: str) -> dict[str, Any]:
    """The edge of the entity the fixture proposes for `table`."""
    found = edges(driver)
    key = next(((e, t) for e, t in found if t == table
                and e.casefold() == PROPOSALS[table].entity.casefold()), None)
    assert key is not None, f"no edge from {PROPOSALS[table].entity} to {table}: {sorted(found)}"
    return found[key]


@pytest.mark.parametrize("table", sorted(COARSER))
def test_an_owner_edge_holds_the_entitys_key_identity_and_grade(
    built: IdentityBuild, driver: Driver, table: str,
) -> None:
    built.built()

    props = edge(driver, table)

    assert props.get("owner") is True
    assert props.get("key_columns") == list(COARSER[table])
    assert props.get("verification") == ("proven" if table == "hull_rev" else "supported")
    identity = EntityIdentity.model_validate_json(props["identity"])
    assert identity.key_columns == COARSER[table]
    by = ordered_by(table)
    assert all(a.by == by for a in identity.attributes)
    assert props.get("null_key_rows") == (null_row_count(table, COARSER[table]) or None)


def test_a_shared_entity_gets_an_edge_with_owner_false(built: IdentityBuild,
                                                       driver: Driver) -> None:
    built.built()

    props = edges(driver)[(PROPOSALS["hull_rev"].entity, "hull_survey")]

    assert props.get("owner") is False
    assert props.get("key_columns") == ["hull_no"]


def test_an_edge_l2_wrote_before_the_step_never_carries_owner_false(
    built: IdentityBuild, driver: Driver,
) -> None:
    built.built()

    props = edges(driver)[(L2_ENTITIES["hull_cargo"], "hull_cargo")]

    assert props.get("owner") is not False
    assert props.get("key_columns") == ["hull_no"], "the step states the shared key on it"


def test_a_relation_whose_values_are_not_contained_gets_no_entity_edge(
    built: IdentityBuild, driver: Driver,
) -> None:
    built.built()
    found = edges(driver)

    assert (PROPOSALS["hull_rev"].entity, "hull_quote") not in found
    assert (PROPOSALS["hull_rev"].entity, "hull_survey") in found


def test_each_coarser_entity_has_its_has_identity_assertion(built: IdentityBuild,
                                                            driver: Driver) -> None:
    built.built()
    payloads = {}
    for r in _rows(driver, _IDENTITIES):
        payload = json.loads(r["payload"]) if isinstance(r["payload"], str) else r["payload"]
        if payload.get("identity"):
            payloads[CanonicalRef.parse(r["subject"]).table] = payload

    assert set(payloads) >= set(COARSER)
    for table, key in COARSER.items():
        payload = payloads[table]
        assert (payload["entity"], payload["key_columns"]) == (PROPOSALS[table].entity, list(key))
        assert set(payload["evidence"]) >= {"rows", "groups", "null_rows", "conflict_share"}
