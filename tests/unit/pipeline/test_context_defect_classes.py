"""Done-when 1, core: the four context defect classes on the product's own query path.

Each class is asserted as the contract states its observable, with its decoy, on the context
the core path renders: retrieval over a stand-in graph, then the context object, then the
SQL prompt; or the context object itself where the class lives in rendering only. The model
is a mock. Graph cases and their questions live in `tests/support/connector_fixture_context`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sema.consumers.base import ConsumerDeps, ConsumerRequest
from sema.consumers.nl2sql.consumer import NL2SQLConsumer
from sema.consumers.nl2sql.prompting import build_sql_prompt
from sema.models.context import GovernedValue, PhysicalAsset, SemanticContextObject
from sema.pipeline.context import prune_to_sco
from tests.support.connector_fixture import CATALOG, MAIN_SCHEMA, MAIN_TABLES
from tests.support.connector_fixture_context import (
    ASKED_HIERARCHY_QUESTION,
    DEFINED_COLUMN,
    DEFINITION_ONLY_QUESTION,
    GOODS_COLUMN,
    GOODS_TERMS,
    GOODS_VOCABULARY,
    LABELLED_REGIONS,
    NAMED_COLUMN,
    NAMED_TERM_QUESTION,
    REGION_COLUMN,
    RESULT_COLUMNS,
    SHAPE_QUESTION,
    SHAPE_SQL,
    UNASKED_HIERARCHY_QUESTION,
    UNLABELLED_REGION,
    VALUES_QUESTION,
    CaseGraph,
    case_engine,
    hierarchy_edge_lines,
    hierarchy_graph,
    lines_naming,
    ranking_graph,
    ranks_before_alternative,
)

pytestmark = pytest.mark.unit

HIERARCHY_MARK = "HIERARCHY"
RESULT_SHAPE_MARK = "RESULT SHAPE"


def _context(graph: CaseGraph, question: str) -> str:
    return build_sql_prompt(prune_to_sco(case_engine(graph).retrieve(question)), question)


def _asset(table: str) -> PhysicalAsset:
    (fixture,) = (t for t in MAIN_TABLES if t.name == table)
    return PhysicalAsset(catalog=CATALOG, schema=MAIN_SCHEMA, table=table,
                         columns=list(fixture.column_names))


def _values_sco() -> SemanticContextObject:
    """A plain coded column with one unlabelled member, and an ambiguous coded column."""
    region_table, region_column = REGION_COLUMN
    goods_table, goods_column = GOODS_COLUMN
    regions = [*LABELLED_REGIONS, (UNLABELLED_REGION, "")]
    goods = [(t.code, t.label) for t in GOODS_TERMS if t.parent]
    return SemanticContextObject(
        physical_assets=[_asset(region_table), _asset(goods_table)],
        governed_values=[
            GovernedValue(property_name=region_column, column=region_column, table=region_table,
                          values=[{"code": c, "label": label} for c, label in regions]),
            GovernedValue(property_name=goods_column, column=goods_column, table=goods_table,
                          values=[{"code": c, "label": label} for c, label in goods],
                          vocabulary=GOODS_VOCABULARY, ambiguous=True),
        ],
    )


def test_d1_a_question_naming_no_hierarchy_concept_gets_no_hierarchy_section() -> None:
    text = _context(hierarchy_graph(), UNASKED_HIERARCHY_QUESTION)

    assert f"{CATALOG}.{MAIN_SCHEMA}.transfer" in text, "the context lists no relation"
    assert hierarchy_edge_lines(text, GOODS_TERMS) == []
    assert HIERARCHY_MARK not in text


def test_d1_decoy_a_question_naming_a_hierarchy_concept_gets_the_section() -> None:
    text = _context(hierarchy_graph(), ASKED_HIERARCHY_QUESTION)

    assert hierarchy_edge_lines(text, GOODS_TERMS), text


def test_d2_every_rendered_governed_value_carries_its_label() -> None:
    text = build_sql_prompt(_values_sco(), VALUES_QUESTION)

    labelled = [*LABELLED_REGIONS, *((t.code, t.label) for t in GOODS_TERMS if t.parent)]
    for code, label in labelled:
        rendered = lines_naming(text, code)
        assert rendered, f"{code} is not rendered"
        assert all(label in line for line in rendered), f"{code} without {label!r}: {rendered}"
    stated = [x for x in text.splitlines() if "answer" in x.lower() and "label" in x.lower()]
    assert stated, "the context does not state that answers show the label"


def test_d2_decoy_a_value_with_no_label_renders_its_code_and_says_so() -> None:
    text = build_sql_prompt(_values_sco(), VALUES_QUESTION)

    rendered = lines_naming(text, UNLABELLED_REGION)
    assert rendered, f"{UNLABELLED_REGION} is not rendered"
    assert all("no label" in line.lower() for line in rendered), rendered


def test_d3_a_named_column_ranks_before_a_column_whose_definition_is_closer() -> None:
    text = _context(ranking_graph(NAMED_TERM_QUESTION), NAMED_TERM_QUESTION)

    ranked = [x for x in text.splitlines()
              if ranks_before_alternative(x, NAMED_COLUMN.column, DEFINED_COLUMN.column)]
    assert ranked, text


def test_d3_decoy_a_term_matching_only_a_definition_still_reaches_that_column() -> None:
    text = _context(ranking_graph(DEFINITION_ONLY_QUESTION), DEFINITION_ONLY_QUESTION)

    assert lines_naming(text, DEFINED_COLUMN.column), text
    demoted = [x for x in text.splitlines()
               if ranks_before_alternative(x, NAMED_COLUMN.column, DEFINED_COLUMN.column)]
    assert demoted == []


def _shaped_sco() -> SemanticContextObject:
    from sema.models.context import ResultShape

    return SemanticContextObject(
        physical_assets=[_asset(REGION_COLUMN[0])],
        result_shape=ResultShape(columns=list(RESULT_COLUMNS)),
    )


def test_d4_the_context_renders_one_result_shape_block_in_column_order() -> None:
    text = build_sql_prompt(_shaped_sco(), SHAPE_QUESTION)

    assert text.count(RESULT_SHAPE_MARK) == 1, text
    block = text[text.index(RESULT_SHAPE_MARK):]
    positions = [block.find(column) for column in RESULT_COLUMNS]
    assert -1 not in positions and positions == sorted(positions), block


def test_d4_decoy_a_context_with_no_shape_fields_renders_no_block() -> None:
    sco = SemanticContextObject(physical_assets=[_asset(REGION_COLUMN[0])])

    assert RESULT_SHAPE_MARK not in build_sql_prompt(sco, SHAPE_QUESTION)


def test_d4_the_consumer_records_the_statement_column_list() -> None:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=SHAPE_SQL)
    sco = SemanticContextObject(physical_assets=[_asset(REGION_COLUMN[0])])

    plan = NL2SQLConsumer().plan(ConsumerRequest(SHAPE_QUESTION, "plan"), sco,
                                 ConsumerDeps(llm=llm))

    assert plan.valid, plan.errors
    assert plan.columns == list(RESULT_COLUMNS)
