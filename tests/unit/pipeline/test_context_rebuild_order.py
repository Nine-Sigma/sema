"""Done-when 2 (N4): the rendered context does not depend on the order the graph returns rows.

The neutral stand-in graph (`tests/support/connector_fixture_context.py`) answers each read the
core query path makes. A second driver over the same graph returns every read's rows in
reverse; a vector read keeps its score order, so only rows of equal score swap, as an index
with another tie order would return them. Same facts and question, so the rendered text must
be byte-equal. A cap that cuts through a tie keeps the same rows in both orders; rows that
differ in score keep their score order in both.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from sema.consumers.nl2sql.prompting import render_sections
from sema.pipeline.context import prune_to_sco
from tests.support.connector_fixture import CATALOG, MAIN_SCHEMA
from tests.support.connector_fixture_context import (
    ASKED_HIERARCHY_QUESTION,
    DEFINITION_ONLY_QUESTION,
    ENTITY_INDEX,
    ENTITY_SCORE,
    JOIN_QUESTION,
    NAMED_TERM_QUESTION,
    UNASKED_HIERARCHY_QUESTION,
    VALUES_QUESTION,
    CaseGraph,
    case_engine,
    entity_node,
    hierarchy_graph,
    join_graph,
    ranking_graph,
)

pytestmark = pytest.mark.unit


class ReturnedReversed:
    """The same graph, returning each read's rows in reverse; scored rows stay in score order."""

    def __init__(self, graph: CaseGraph) -> None:
        self.graph = graph

    def session(self, **_: Any) -> ReturnedReversed:
        return self

    def __enter__(self) -> ReturnedReversed:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def run(self, query: str, **params: Any) -> Iterator[Any]:
        rows = list(self.graph.run(query, **params))[::-1]
        if rows and "score" in rows[0].data():
            rows.sort(key=lambda row: -row.data()["score"])
        return iter(rows)


def _tied_tables() -> CaseGraph:
    return CaseGraph({ENTITY_INDEX: [(entity_node("customer"), ENTITY_SCORE),
                                     (entity_node("account"), ENTITY_SCORE)]})


CASES: dict[str, tuple[str, Callable[[], CaseGraph], str]] = {
    "tables_and_governed_values": (VALUES_QUESTION, _tied_tables, "GOVERNED FILTER VALUES"),
    "joins": (JOIN_QUESTION, join_graph, "JOIN PATHS"),
    "ranking_named_term": (NAMED_TERM_QUESTION, lambda: ranking_graph(NAMED_TERM_QUESTION),
                           "alternative"),
    "ranking_definition_only": (DEFINITION_ONLY_QUESTION,
                                lambda: ranking_graph(DEFINITION_ONLY_QUESTION),
                                "AVAILABLE TABLES AND COLUMNS"),
    "hierarchy_asked": (ASKED_HIERARCHY_QUESTION, hierarchy_graph, "Hand tools"),
    "hierarchy_unasked": (UNASKED_HIERARCHY_QUESTION, hierarchy_graph,
                          "AVAILABLE TABLES AND COLUMNS"),
}


def render(driver: Any, question: str, **caps: int) -> str:
    sco = prune_to_sco(case_engine(driver).retrieve(question), **caps)
    return "\n".join(render_sections(sco, "duckdb"))


def table_lines(text: str) -> list[str]:
    prefix = f"{CATALOG}.{MAIN_SCHEMA}."
    return [line.strip().split(":")[0] for line in text.splitlines()
            if line.startswith(f"  {prefix}") and "->" not in line]


def join_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if " -> " in line]


@pytest.mark.parametrize("case", list(CASES))
def test_the_context_is_byte_equal_whatever_order_the_graph_returns_rows(case: str) -> None:
    question, graph, shown = CASES[case]

    forward = render(graph(), question)
    backward = render(ReturnedReversed(graph()), question)

    assert shown in forward, forward
    assert forward == backward


def test_a_table_cap_through_a_tie_keeps_the_same_table_in_both_orders() -> None:
    forward = render(_tied_tables(), VALUES_QUESTION, max_entities=1)
    backward = render(ReturnedReversed(_tied_tables()), VALUES_QUESTION, max_entities=1)

    assert len(table_lines(forward)) == 1, forward
    assert table_lines(forward) == table_lines(backward)
    assert forward == backward


def test_a_join_cap_through_a_tie_keeps_the_same_join_in_both_orders() -> None:
    forward = render(join_graph(), JOIN_QUESTION, max_joins=1)
    backward = render(ReturnedReversed(join_graph()), JOIN_QUESTION, max_joins=1)

    assert len(join_lines(forward)) == 1, forward
    assert join_lines(forward) == join_lines(backward)
    assert forward == backward


def test_decoy_tables_that_differ_in_score_keep_their_score_order_in_both_orders() -> None:
    def scored() -> CaseGraph:
        return CaseGraph({ENTITY_INDEX: [(entity_node("customer"), 0.8),
                                         (entity_node("account"), 0.6)]})

    higher, lower = (f"{CATALOG}.{MAIN_SCHEMA}.{t}" for t in ("customer", "account"))
    for driver in (scored(), ReturnedReversed(scored())):
        assert table_lines(render(driver, VALUES_QUESTION)) == [higher, lower]
        assert table_lines(render(driver, VALUES_QUESTION, max_entities=1)) == [higher]
