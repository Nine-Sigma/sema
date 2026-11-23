"""Done-when 2 (N4) on a real graph: two writes of one neutral graph render one context.

The neutral fixture's relations, entities, properties, governed values, goods hierarchy and
detected joins (`tests/support/connector_fixture_context.py`) are written to Neo4j one node
and one edge per statement, rendered for each neutral question on the core query path, cleared,
written again with the statements in reverse order, and rendered again. The test controls the
write order itself; the contexts must be byte-equal.
"""

from __future__ import annotations

from typing import Any

import pytest

from sema.consumers.nl2sql.prompting import render_sections
from sema.pipeline.context import prune_to_sco
from sema.pipeline.retrieval import RetrievalEngine
from tests.support.connector_fixture import (
    CATALOG,
    EXPECTED_DETECTED_JOINS,
    MAIN_SCHEMA,
    MAIN_TABLES,
)
from tests.support.connector_fixture_context import (
    ASKED_HIERARCHY_QUESTION,
    CATEGORICAL,
    DEFINITION_ONLY_QUESTION,
    GOODS_COLUMN,
    GOODS_TERMS,
    GOODS_VOCABULARY,
    LABELLED_REGIONS,
    REGION_COLUMN,
    UNLABELLED_REGION,
    entity_name,
    entity_node,
    join_row,
    property_node,
)

pytestmark = pytest.mark.integration

Statement = tuple[str, dict[str, Any]]

QUESTIONS = {
    "Which customer account made a transfer?": "JOIN PATHS",
    "Which order line holds a product?": "GOVERNED FILTER VALUES",
    DEFINITION_ONLY_QUESTION: "JOIN PATHS",
    ASKED_HIERARCHY_QUESTION: "TERM HIERARCHY",
}


def _ref(table: str, column: str | None = None) -> str:
    return ".".join(p for p in (CATALOG, MAIN_SCHEMA, table, column) if p)


def _value_set(table: str, column: str) -> str:
    return f"{table}_{column}_values"


def _members() -> dict[tuple[str, str], list[str]]:
    regions = [code for code, _ in LABELLED_REGIONS] + [UNLABELLED_REGION]
    return {GOODS_COLUMN: [t.code for t in GOODS_TERMS if t.parent], REGION_COLUMN: regions}


def _term_nodes() -> list[dict[str, Any]]:
    goods = [{"code": t.code, "label": t.label, "vocabulary_name": GOODS_VOCABULARY}
             for t in GOODS_TERMS]
    regions = [{"code": code, "label": label}
               for code, label in (*LABELLED_REGIONS, (UNLABELLED_REGION, ""))]
    return [{**term, "confidence": 0.9, "status": "auto"} for term in (*goods, *regions)]


def _table_nodes() -> list[Statement]:
    nodes: list[Statement] = []
    for table in MAIN_TABLES:
        nodes.append(("CREATE (:Table $props)", {"props": {
            "name": table.name, "schema_name": MAIN_SCHEMA, "catalog": CATALOG,
            "ref": _ref(table.name)}}))
        nodes.append(("CREATE (:Entity $props)", {"props": entity_node(table.name)}))
        for column in table.column_names:
            categorical = (table.name, column) in CATEGORICAL
            nodes.append(("CREATE (:Column $props)", {"props": {
                "name": column, "table_name": table.name, "schema_name": MAIN_SCHEMA,
                "catalog": CATALOG, "data_type": "string", "ref": _ref(table.name, column)}}))
            nodes.append(("CREATE (:Property $props)", {"props": {
                **property_node(table.name, column),
                "semantic_type": "categorical" if categorical else "free_text"}}))
    return nodes


def graph_nodes() -> list[Statement]:
    nodes = _table_nodes()
    nodes += [("CREATE (:ValueSet $props)", {"props": {
        "name": _value_set(*column), "column_ref": _ref(*column)}}) for column in _members()]
    nodes += [("CREATE (:Term $props)", {"props": term}) for term in _term_nodes()]
    for join in EXPECTED_DETECTED_JOINS:
        row = {k: v for k, v in join_row(join).items() if v is not None}
        nodes.append(("CREATE (:JoinPath $props)", {"props": row}))
    return nodes


def _table_edges() -> list[Statement]:
    edges: list[Statement] = []
    for table in MAIN_TABLES:
        entity = entity_name(table.name)
        edges.append(("MATCH (e:Entity {name: $e}), (t:Table {ref: $t}) "
                      "CREATE (e)-[:ENTITY_ON_TABLE]->(t)", {"e": entity, "t": _ref(table.name)}))
        for column in table.column_names:
            prop = property_node(table.name, column)["name"]
            ids = {"e": entity, "p": prop, "c": _ref(table.name, column), "t": _ref(table.name)}
            edges.append(("MATCH (c:Column {ref: $c}), (t:Table {ref: $t}) "
                          "CREATE (c)-[:IN_TABLE]->(t)", ids))
            edges.append(("MATCH (e:Entity {name: $e}), (p:Property {entity_name: $e, name: $p}) "
                          "CREATE (e)-[:HAS_PROPERTY]->(p)", ids))
            edges.append(("MATCH (p:Property {entity_name: $e, name: $p}), "
                          "(c:Column {ref: $c}) CREATE (p)-[:PROPERTY_ON_COLUMN]->(c)", ids))
    return edges


def graph_edges() -> list[Statement]:
    edges = _table_edges()
    for column, codes in _members().items():
        ids = {"c": _ref(*column), "vs": _value_set(*column)}
        edges.append(("MATCH (c:Column {ref: $c}), (vs:ValueSet {name: $vs}) "
                      "CREATE (c)-[:HAS_VALUE_SET]->(vs)", ids))
        edges += [("MATCH (t:Term {code: $code}), (vs:ValueSet {name: $vs}) "
                   "CREATE (t)-[:MEMBER_OF]->(vs)", {**ids, "code": code}) for code in codes]
    edges += [("MATCH (p:Term {code: $parent}), (c:Term {code: $child}) "
               "CREATE (p)-[:PARENT_OF]->(c)", {"parent": t.parent, "child": t.code})
              for t in GOODS_TERMS if t.parent]
    for join in EXPECTED_DETECTED_JOINS:
        for table in (join.referencing_table, join.referenced_table):
            edges.append(("MATCH (jp:JoinPath {name: $jp}), (t:Table {ref: $t}) "
                          "CREATE (jp)-[:USES]->(t)",
                          {"jp": join_row(join)["name"], "t": _ref(str(table))}))
    return edges


def write(driver: Any, statements: list[Statement]) -> None:
    with driver.session() as session:
        for query, params in statements:
            session.run(query, **params).consume()


def contexts(driver: Any) -> dict[str, str]:
    engine = RetrievalEngine(driver=driver)
    return {question: "\n".join(render_sections(prune_to_sco(engine.retrieve(question)), "duckdb"))
            for question in QUESTIONS}


def test_a_graph_written_in_reverse_order_renders_byte_equal_contexts(clean_neo4j: Any) -> None:
    nodes, edges = graph_nodes(), graph_edges()

    write(clean_neo4j, nodes + edges)
    first = contexts(clean_neo4j)
    write(clean_neo4j, [("MATCH (n) DETACH DELETE n", {})])
    write(clean_neo4j, nodes[::-1] + edges[::-1])
    second = contexts(clean_neo4j)

    assert {q: shown in first[q] for q, shown in QUESTIONS.items()} == dict.fromkeys(QUESTIONS, True)
    assert {q: first[q] == second[q] for q in QUESTIONS} == dict.fromkeys(QUESTIONS, True), (
        first, second)
