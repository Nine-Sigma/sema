"""Done-when item 1: the join graph `run_build` writes from the neutral fixture.

One test per clause, so a red run names each clause that fails.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import SecretStr

from sema.models.config import Neo4jConfig
from sema.models.context import SemanticCandidateSet
from sema.pipeline.context_utils import _build_join_paths
from sema.pipeline.orchestrate import run_build
from sema.pipeline.retrieval import RetrievalEngine
from sema.pipeline.retrieval_utils import _expand_joins
from tests.support.connector_fixture import (
    DECOY_JOINS,
    DECOY_SCHEMA,
    DECOY_TABLES,
    EXPECTED_DETECTED_JOINS,
    MAIN_SCHEMA,
    MAIN_TABLES,
    FixtureJoin,
    fixture_build_config,
    fixture_environment,
)

pytestmark = pytest.mark.integration

Pair = tuple[str, str, str, str]

COMPOSITE_KEY: list[Pair] = [
    ("order_line", "order_id", "shipment_line", "order_id"),
    ("order_line", "line_no", "shipment_line", "line_no"),
]
DECLARED_ROLES: list[list[Pair]] = [
    [("account", "account_id", "transfer", "from_account_id")],
    [("account", "account_id", "transfer", "to_account_id")],
]
DECLARED_AND_DETECTED: list[Pair] = [("customer", "customer_id", "account", "customer_id")]


@dataclass(frozen=True)
class JoinNode:
    node_id: str
    name: str
    pairs: list[Pair]
    uses: set[tuple[str, str]]
    from_entities: set[str]
    to_entities: set[str]


@dataclass(frozen=True)
class FixtureGraph:
    joins: list[JoinNode]
    entities_on_table: dict[tuple[str, str], set[str]]
    declared_payloads: list[list[Pair]]
    driver: Any

    def with_pairs(self, pairs: list[Pair]) -> list[JoinNode]:
        return [join for join in self.joins if join.pairs == pairs]


def _pairs(raw: Any) -> list[Pair]:
    items = json.loads(raw) if isinstance(raw, str) else (raw or [])
    return [
        (p.get("left_table", ""), p.get("left_column", ""),
         p.get("right_table", ""), p.get("right_column", ""))
        for p in items if isinstance(p, dict)
    ]


def _payload(raw: Any) -> dict[str, Any]:
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    return parsed if isinstance(parsed, dict) else {}


def _read(driver: Any, query: str) -> list[dict[str, Any]]:
    with driver.session() as session:
        return [record.data() for record in session.run(query)]


_JOINS_QUERY = (
    "MATCH (jp:JoinPath) "
    "OPTIONAL MATCH (jp)-[:USES]->(t:Table) "
    "WITH jp, collect(DISTINCT [t.name, t.schema_name]) AS uses "
    "OPTIONAL MATCH (jp)-[:FROM_ENTITY]->(fe:Entity) "
    "WITH jp, uses, collect(DISTINCT elementId(fe)) AS from_entities "
    "OPTIONAL MATCH (jp)-[:TO_ENTITY]->(te:Entity) "
    "RETURN elementId(jp) AS node_id, jp.name AS name, "
    "jp.join_predicates AS join_predicates, uses, from_entities, "
    "collect(DISTINCT elementId(te)) AS to_entities"
)
_ENTITIES_QUERY = (
    "MATCH (e:Entity)-[:ENTITY_ON_TABLE]->(t:Table) "
    "RETURN elementId(e) AS entity, t.name AS table_name, t.schema_name AS schema_name"
)
_DECLARED_QUERY = "MATCH (a:Assertion {predicate: 'joins_to'}) RETURN a.payload AS payload"


def _read_graph(driver: Any) -> FixtureGraph:
    joins = [
        JoinNode(
            node_id=row["node_id"], name=row["name"] or "",
            pairs=_pairs(row["join_predicates"]),
            uses={(name, schema) for name, schema in row["uses"] if name is not None},
            from_entities=set(row["from_entities"]), to_entities=set(row["to_entities"]),
        )
        for row in _read(driver, _JOINS_QUERY)
    ]
    entities: dict[tuple[str, str], set[str]] = {}
    for row in _read(driver, _ENTITIES_QUERY):
        entities.setdefault((row["table_name"], row["schema_name"]), set()).add(row["entity"])
    declared = [
        _pairs(_payload(row["payload"]).get("join_predicates"))
        for row in _read(driver, _DECLARED_QUERY)
    ]
    return FixtureGraph(joins, entities, declared, driver)


def _neo4j_config() -> Neo4jConfig:
    return Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=SecretStr(os.getenv("NEO4J_PASSWORD", "graphrag")),
    )


@pytest.fixture
def graph(clean_neo4j: Any) -> FixtureGraph:
    with fixture_environment():
        report = run_build(fixture_build_config(_neo4j_config()))
    assert report["tables_failed"] == 0, report["failed_tables"]
    built = _read_graph(clean_neo4j)
    assert built.joins, "run_build wrote no JoinPath"
    return built


def _matches_decoy(join: JoinNode, decoy: FixtureJoin) -> bool:
    if len(join.pairs) != 1:
        return False
    left_table, left_column, right_table, right_column = join.pairs[0]
    if (right_table, right_column) != (decoy.referencing_table, decoy.referencing_column):
        return False
    if decoy.referenced_table is None:
        return True
    return (left_table, left_column) == (decoy.referenced_table, decoy.referenced_column)


def test_every_join_uses_each_relation_its_predicates_name(graph: FixtureGraph) -> None:
    missing = {
        join.name: sorted(needed - join.uses)
        for join in graph.joins
        if (needed := {(t, MAIN_SCHEMA) for p in join.pairs for t in (p[0], p[2])}) - join.uses
    }
    assert not missing, sorted(missing)


def test_no_join_uses_the_same_named_decoy_table(graph: FixtureGraph) -> None:
    decoy_names = {table.name for table in DECOY_TABLES}
    naming_decoy = [
        join for join in graph.joins
        if decoy_names & {table for p in join.pairs for table in (p[0], p[2])}
    ]
    assert naming_decoy, "no join names the decoy table's name"
    for join in naming_decoy:
        assert join.uses & {(name, MAIN_SCHEMA) for name in decoy_names}, f"{join.name}: no USES"
        assert DECOY_SCHEMA not in {schema for _, schema in join.uses}, f"{join.name}: decoy USES"


def test_two_column_declared_key_is_one_join_with_ordered_predicates(
    graph: FixtureGraph,
) -> None:
    assert len(graph.with_pairs(COMPOSITE_KEY)) == 1


def test_two_declared_roles_are_two_joins(graph: FixtureGraph) -> None:
    found = [graph.with_pairs(pairs) for pairs in DECLARED_ROLES]
    assert [len(joins) for joins in found] == [1, 1]
    assert found[0][0].node_id != found[1][0].node_id


def test_no_join_has_an_empty_name_or_no_predicates(graph: FixtureGraph) -> None:
    empty = [(join.name, join.pairs) for join in graph.joins if not join.name or not join.pairs]
    assert not empty


def test_declared_and_detected_join_on_one_column_pair_is_one_join(
    graph: FixtureGraph,
) -> None:
    assert DECLARED_AND_DETECTED in graph.declared_payloads
    assert len(graph.with_pairs(DECLARED_AND_DETECTED)) == 1


def test_each_join_between_entities_links_both_entities(graph: FixtureGraph) -> None:
    linked = 0
    for join in graph.joins:
        if not join.pairs:
            continue
        left, right = join.pairs[0][0], join.pairs[0][2]
        left_entities = graph.entities_on_table.get((left, MAIN_SCHEMA), set())
        right_entities = graph.entities_on_table.get((right, MAIN_SCHEMA), set())
        if not left_entities or not right_entities:
            continue
        assert join.from_entities & left_entities, f"{join.name}: no FROM_ENTITY"
        assert join.to_entities & right_entities, f"{join.name}: no TO_ENTITY"
        linked += 1
    assert linked, "no join has an entity on both sides"


def test_context_join_paths_name_both_tables(graph: FixtureGraph) -> None:
    engine = RetrievalEngine(driver=graph.driver)
    rows = _expand_joins(engine, [table.name for table in MAIN_TABLES])
    assert {row["name"] for row in rows} == {join.name for join in graph.joins}
    candidates = SemanticCandidateSet(
        query="", candidates=[{"type": "join", **row} for row in rows],
    )
    paths = _build_join_paths(candidates, max_joins=len(rows))
    assert [(p.from_table, p.to_table) for p in paths if not p.from_table or not p.to_table] == []


def test_every_expected_detected_join_is_present(graph: FixtureGraph) -> None:
    missing = [
        expected for expected in EXPECTED_DETECTED_JOINS
        if not graph.with_pairs([(
            expected.referenced_table or "", expected.referenced_column or "",
            expected.referencing_table, expected.referencing_column,
        )])
    ]
    assert not missing, [f"{m.referencing_table}.{m.referencing_column}" for m in missing]


def test_no_decoy_join_is_present(graph: FixtureGraph) -> None:
    present = [
        (decoy, join.name)
        for decoy in DECOY_JOINS for join in graph.joins if _matches_decoy(join, decoy)
    ]
    assert not present
