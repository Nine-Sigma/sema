"""Done-when 1, arm S: the four context defect classes on the benchmark's S renderer.

The same observables and decoys as the core file, asserted on the context `SContextProvider`
renders from its own inputs: a retriever (the core `RetrievalEngine` over a stand-in graph,
or a fixed candidate list), the sidecar, product specs, bindings and library facts; and, for
the result shape, `render` over a context object. The model is never called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sema.models.context import PhysicalAsset, SemanticContextObject
from showcase.ontology_query_benchmark.sema_bindings import (
    Sidecar,
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.ontology_query_benchmark.sema_build_utils import LibraryFacts
from showcase.ontology_query_benchmark.sema_context import Retriever, SContextProvider
from showcase.ontology_query_benchmark.sema_context_utils import Rendering, render
from tests.support.connector_fixture import CATALOG, MAIN_SCHEMA, MAIN_TABLES
from tests.support.connector_fixture_context import (
    DEFINED_COLUMN,
    DEFINITION_ONLY_QUESTION,
    NAMED_COLUMN,
    NAMED_TERM_QUESTION,
    RESULT_COLUMNS,
    SHAPE_QUESTION,
    case_engine,
    lines_naming,
    ranking_graph,
    ranks_before_alternative,
)
from tests.unit.ontology_query_benchmark.hierarchy_fixture import hierarchy
from tests.unit.ontology_query_benchmark.sema_bindings_env import (
    CONVENTIONS,
    EVENTS,
    catalog_fixture,
    facts_fixture,
    write_fixture,
)
from tests.unit.ontology_query_benchmark.sema_context_env import (
    EVENTS_ENTITY,
    LIMITS,
    PEOPLE_ENTITY,
    PEOPLE_VALUES,
    QUESTION,
    _Retriever,
)
from tests.unit.ontology_query_benchmark.sema_hierarchy_anchor_env import READABLE, anchor
from tests.unit.ontology_query_benchmark.sema_hierarchy_env import _facts

pytestmark = pytest.mark.unit

HIERARCHY_MARK = "HIERARCHY"
S_HIERARCHY_SECTION = "CONCEPT HIERARCHY"
RESULT_SHAPE_MARK = "RESULT SHAPE"
UNASKED_CONCEPT_QUESTION = "How many events were recorded?"
ASKED_CONCEPT_QUESTION = "How many events are a lung neoplasm?"
SIMILAR_ANCHOR = {**anchor("C1", "Neoplasm", "std", 2), "score": 0.41}


def _sidecar(tmp_path: Path) -> Sidecar:
    return load_sidecar(write_fixture(tmp_path), tmp_path)


def _s_text(tmp_path: Path, retriever: Retriever, question: str, *,
            facts: LibraryFacts | None = None, readable: dict[str, bool] | None = None) -> str:
    sidecar = _sidecar(tmp_path)
    products = load_products(sidecar)
    provider = SContextProvider(
        retriever, sidecar, products, resolve_bindings(sidecar, products, catalog_fixture()),
        facts or facts_fixture(tmp_path), graph_identity="graph-1", conventions=CONVENTIONS,
        limits=LIMITS, hierarchy_relations=readable,
    )
    return provider(question).text


def _hierarchy_text(tmp_path: Path, question: str) -> str:
    return _s_text(tmp_path, _Retriever([EVENTS_ENTITY, SIMILAR_ANCHOR]), question,
                   facts=_facts(tmp_path, hierarchy()), readable=READABLE)


def test_d1_a_question_naming_no_hierarchy_concept_gets_no_hierarchy_section(
    tmp_path: Path,
) -> None:
    text = _hierarchy_text(tmp_path, UNASKED_CONCEPT_QUESTION)

    assert f"cat.sch.{EVENTS}" in text, "the context lists no relation"
    assert HIERARCHY_MARK not in text
    assert _sidecar(tmp_path).hierarchy_heading not in text


def test_d1_decoy_a_question_naming_a_hierarchy_concept_gets_the_section(
    tmp_path: Path,
) -> None:
    assert S_HIERARCHY_SECTION in _hierarchy_text(tmp_path, ASKED_CONCEPT_QUESTION)


def _value(candidate: dict[str, Any], label: str) -> dict[str, Any]:
    return {**candidate, "label": label}


def test_d2_every_rendered_governed_value_carries_its_label(tmp_path: Path) -> None:
    text = _s_text(tmp_path, _Retriever([PEOPLE_ENTITY, *PEOPLE_VALUES]), QUESTION)

    for value in PEOPLE_VALUES:
        rendered = lines_naming(text, value["code"])
        assert rendered, f"{value['code']} is not rendered"
        assert all(value["label"] in line for line in rendered), rendered
    stated = [x for x in text.splitlines() if "answer" in x.lower() and "label" in x.lower()]
    assert stated, "the context does not state that answers show the label"


def test_d2_decoy_a_value_with_no_label_renders_its_code_and_says_so(tmp_path: Path) -> None:
    *labelled, unlabelled = PEOPLE_VALUES
    values = [*labelled, _value(unlabelled, "")]
    text = _s_text(tmp_path, _Retriever([PEOPLE_ENTITY, *values]), QUESTION)

    rendered = lines_naming(text, unlabelled["code"])
    assert rendered, f"{unlabelled['code']} is not rendered"
    assert all("no label" in line.lower() for line in rendered), rendered


def test_d3_a_named_column_ranks_before_a_column_whose_definition_is_closer(
    tmp_path: Path,
) -> None:
    engine = case_engine(ranking_graph(NAMED_TERM_QUESTION))
    text = _s_text(tmp_path, engine, NAMED_TERM_QUESTION)

    ranked = [x for x in text.splitlines()
              if ranks_before_alternative(x, NAMED_COLUMN.column, DEFINED_COLUMN.column)]
    assert ranked, text


def test_d3_decoy_a_term_matching_only_a_definition_still_reaches_that_column(
    tmp_path: Path,
) -> None:
    engine = case_engine(ranking_graph(DEFINITION_ONLY_QUESTION))
    text = _s_text(tmp_path, engine, DEFINITION_ONLY_QUESTION)

    assert lines_naming(text, DEFINED_COLUMN.column), text
    demoted = [x for x in text.splitlines()
               if ranks_before_alternative(x, NAMED_COLUMN.column, DEFINED_COLUMN.column)]
    assert demoted == []


def _customer() -> PhysicalAsset:
    (fixture,) = (t for t in MAIN_TABLES if t.name == "customer")
    return PhysicalAsset(catalog=CATALOG, schema=MAIN_SCHEMA, table=fixture.name,
                         columns=list(fixture.column_names))


def _rendered(tmp_path: Path, sco: SemanticContextObject) -> str:
    sidecar = _sidecar(tmp_path)
    products = load_products(sidecar)
    resolution = resolve_bindings(sidecar, products, catalog_fixture())
    return render(Rendering(sco), sidecar, products, resolution, facts_fixture(tmp_path),
                  question=SHAPE_QUESTION).text


def test_d4_the_context_renders_one_result_shape_block_in_column_order(tmp_path: Path) -> None:
    from sema.models.context import ResultShape

    sco = SemanticContextObject(physical_assets=[_customer()],
                                result_shape=ResultShape(columns=list(RESULT_COLUMNS)))
    text = _rendered(tmp_path, sco)

    assert text.count(RESULT_SHAPE_MARK) == 1, text
    block = text[text.index(RESULT_SHAPE_MARK):]
    positions = [block.find(column) for column in RESULT_COLUMNS]
    assert -1 not in positions and positions == sorted(positions), block


def test_d4_decoy_a_context_with_no_shape_fields_renders_no_block(tmp_path: Path) -> None:
    text = _rendered(tmp_path, SemanticContextObject(physical_assets=[_customer()]))

    assert RESULT_SHAPE_MARK not in text
