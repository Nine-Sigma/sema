"""The hierarchy anchor query against a real graph written by the core loaders."""

import pytest

from sema.graph.cross_layer_bridge import write_value_bridge
from sema.graph.loader import GraphLoader
from sema.graph.queries import CypherQueries

pytestmark = pytest.mark.integration

SCHEMA = "fleet_src"


@pytest.fixture
def anchored_graph(clean_neo4j):
    loader = GraphLoader(clean_neo4j)
    loader.upsert_value_set(
        "kind_values", column_name="kind_code", table_name="fleet",
        schema_name=SCHEMA, catalog="c", source_schema=SCHEMA,
    )
    for code, label in (("S100", "Vehicle"), ("S110", "Car"), ("S120", "Truck"), ("S900", "Bicycle")):
        loader.upsert_term(code, label, source="test", confidence=0.9,
                           source_schema=SCHEMA, vocabulary_name="STD")
    for child in ("S110", "S120"):
        loader.add_term_hierarchy("S100", child, source_schema=SCHEMA, vocabulary_name="STD")
    for src, concept in (("c1", "S110"), ("c2", "S120"), ("c3", "S900")):
        loader.add_term_to_value_set(src, "kind_values", source_schema=SCHEMA,
                                     vocabulary_name="SRC", value_set_ref=f"c.{SCHEMA}.fleet.kind_code")
        write_value_bridge(loader, source_vocabulary="SRC", source_code=src,
                           concept_vocabulary="STD", concept_code=concept, source_schema=SCHEMA)
    return clean_neo4j


def run(driver, code, member_cap=200):
    with driver.session() as session:
        return [r.data() for r in session.run(
            CypherQueries.expand_hierarchy_anchor(max_depth=3),
            code=code, vocabulary_name="STD", member_cap=member_cap,
        )]


def test_anchor_reaches_the_column_through_bridged_members(anchored_graph):
    rows = run(anchored_graph, "S100")

    assert [(r["table_name"], r["column_name"], r["member_count"], r["member_codes"]) for r in rows] == [
        ("fleet", "kind_code", 2, ["c1", "c2"]),
    ]


def test_member_cap_limits_codes_not_count(anchored_graph):
    rows = run(anchored_graph, "S100", member_cap=1)

    assert (rows[0]["member_count"], rows[0]["member_codes"]) == (2, ["c1"])


def test_leaf_term_is_not_an_anchor(anchored_graph):
    assert run(anchored_graph, "S900") == []
