"""Each governed column shows the anchors whose label the question covers first, at most a
named number of them, always beside the {anchor_name} template; each anchor literal is
quoted in the warehouse's own SQL dialect."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from showcase.ontology_query_benchmark.sema_bindings import (
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.ontology_query_benchmark.sema_build_utils import LibraryFacts
from showcase.ontology_query_benchmark.sema_context import SContextProvider
from showcase.ontology_query_benchmark.sema_hierarchy_sql import ANCHORS_PER_COLUMN
from tests.unit.ontology_query_benchmark import test_sema_hierarchy_acris as acris
from tests.unit.ontology_query_benchmark.hierarchy_fixture import hierarchy, warehouse
from tests.unit.ontology_query_benchmark.test_sema_bindings import (
    CONVENTIONS,
    catalog_fixture,
    write_fixture,
)
from tests.unit.ontology_query_benchmark.test_sema_context import (
    EVENTS_ENTITY,
    LIMITS,
    _Retriever,
)
from tests.unit.ontology_query_benchmark.test_sema_hierarchy import RELATIONS, _facts
from tests.unit.ontology_query_benchmark.test_sema_hierarchy_anchor import (
    EVENTS_SQL,
    anchor,
    members,
)

pytestmark = pytest.mark.unit

ACRIS_SQL = "SELECT doc_type AS site_S1 FROM wh.acris.master"
CRAFTED = "x' OR TRUE) --"


def _text(
    tmp_path: Path, facts: LibraryFacts, relations: tuple[str, ...],
    anchors: list[dict[str, Any]], question: str, dialect: str = "duckdb",
) -> str:  # fmt: skip
    sidecar = load_sidecar(write_fixture(tmp_path, sql_dialect=dialect), tmp_path)
    products = load_products(sidecar)
    provider = SContextProvider(
        _Retriever([EVENTS_ENTITY, *anchors]), sidecar, products,
        resolve_bindings(sidecar, products, catalog_fixture()), facts,
        graph_identity="graph-1", conventions=CONVENTIONS, limits=LIMITS,
        hierarchy_relations=dict.fromkeys(relations, True),
    )
    return provider(question).text


def _anchors(text: str) -> list[str]:
    return [found.group(1) for x in text.splitlines()
            if (found := re.match(r"anchor name '(.*?)'(?:, vocabulary|;)", x.strip()))]


def _filled(text: str) -> list[str]:
    return [x.strip().removeprefix("WHERE ") for x in text.splitlines()
            if x.strip().startswith("WHERE ") and "{anchor_name}" not in x]  # fmt: skip


def _acris_text(tmp_path: Path, question: str) -> tuple[Any, str]:
    con = acris._acris_warehouse()
    found = [anchor("OTHER DOCUMENTS", "OTHER DOCUMENTS", "lib", 40),
             anchor(acris.CLASS, acris.CLASS, "lib", 23)]  # fmt: skip
    return con, _text(tmp_path, acris._facts(tmp_path, con), acris.RELATIONS, found, question)


def test_the_anchor_whose_label_the_question_covers_ranks_first(tmp_path: Path) -> None:
    con, text = _acris_text(tmp_path, "How many documents are mortgages and instruments?")

    assert _anchors(text) == [acris.CLASS, "OTHER DOCUMENTS"]
    expected = {r["DOC. TYPE"] for r in acris.ROWS if r["CLASS CODE DESCRIPTION"] == acris.CLASS}
    assert members(con, ACRIS_SQL, _filled(text)[0]) == expected and len(expected) == 23


def test_a_question_naming_no_concept_keeps_the_instruction_and_the_template(
    tmp_path: Path,
) -> None:
    _, text = _acris_text(tmp_path, "How many records were filed?")

    assert _anchors(text) == [acris.CLASS, "OTHER DOCUMENTS"]
    assert "Replace {column} with a value column below, {anchor_name} with the anchor" in text
    assert any("{anchor_name}" in x for x in text.splitlines() if x.strip().startswith("WHERE "))


def test_at_most_the_cap_of_filled_anchors_render_and_the_rest_are_counted(
    tmp_path: Path,
) -> None:
    codes = ["C1", "C2", "C3", "C5", "C7"]
    found = [anchor(code, f"Concept {code}", "std", 1) for code in codes]
    text = _text(tmp_path, _facts(tmp_path, hierarchy()), RELATIONS, found, "Which events?")

    assert ANCHORS_PER_COLUMN == 3
    assert _anchors(text) == [f"Concept {code}" for code in codes[:3]]
    assert "2 more retrieved anchors on site_S1 are not shown" in text


def test_a_crafted_anchor_code_selects_only_true_members_in_duckdb(tmp_path: Path) -> None:
    text = _text(tmp_path, _facts(tmp_path, hierarchy()), RELATIONS,
                 [anchor("C1", CRAFTED, "std", 0)], "Which events are crafted?")  # fmt: skip

    [fragment] = _filled(text)
    assert members(warehouse(), EVENTS_SQL, fragment) == set()


def test_a_databricks_warehouse_gets_its_own_string_quoting(tmp_path: Path) -> None:
    text = _text(tmp_path, _facts(tmp_path, hierarchy()), RELATIONS,
                 [anchor("C1", CRAFTED, "std", 0)], "Which events are crafted?",
                 dialect="databricks")  # fmt: skip

    [fragment] = _filled(text)
    assert "= 'x\\' OR TRUE) --'" in fragment


def test_equal_coverage_ranks_by_match_score_then_code(tmp_path: Path) -> None:
    found = [{**anchor(code, f"Concept {code}", "std", 1), "score": score}
             for code, score in (("C5", 0.4), ("C3", 0.9), ("C2", 0.4))]  # fmt: skip
    text = _text(tmp_path, _facts(tmp_path, hierarchy()), RELATIONS, found, "Which events?")

    assert _anchors(text) == ["Concept C3", "Concept C2", "Concept C5"]


def test_the_member_count_reads_as_strict_descendants_only(tmp_path: Path) -> None:
    text = _text(tmp_path, _facts(tmp_path, hierarchy()), RELATIONS,
                 [anchor("C1", "Neoplasm", "std", 2)], "Which events?")  # fmt: skip

    assert ("An anchor's member count is the column's values strictly below it; a value "
            "mapped to the anchor itself is not counted.") in text  # fmt: skip
    assert "anchor name 'Neoplasm', vocabulary 'std'; 2 members" in text


def test_the_section_states_the_join_once_however_many_anchors_are_retrieved(
    tmp_path: Path,
) -> None:
    found = [anchor(code, f"Concept {code}", "std", 1) for code in ("C1", "C2", "C3", "C5")]
    text = _text(tmp_path, _facts(tmp_path, hierarchy()), RELATIONS, found, "Which events?")

    section = text[text.index("CONCEPT HIERARCHY"):]
    predicates = [x for x in section.splitlines() if x.strip().startswith("WHERE ")]
    assert len(predicates) == 2
    assert "{column}" in predicates[0] and "{anchor_name}" in predicates[0]
    assert _filled(text) == [predicates[1].strip().removeprefix("WHERE ")]


def test_a_listed_anchor_name_is_a_literal_in_the_warehouse_dialect(tmp_path: Path) -> None:
    found = [anchor("C0", CRAFTED, "std", 0), anchor("C1", "Neoplasm", "std", 2)]
    text = _text(tmp_path, _facts(tmp_path, hierarchy()), RELATIONS, found,
                 "Which neoplasm events?", dialect="databricks")  # fmt: skip

    assert "anchor name 'x\\' OR TRUE) --', vocabulary 'std'; 0 members" in text
    assert "anchor name 'Neoplasm', vocabulary 'std'; 2 members" in text
