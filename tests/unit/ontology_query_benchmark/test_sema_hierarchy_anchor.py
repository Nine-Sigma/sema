"""A retrieved hierarchy anchor fills arm S's fragment, the slice sets the walk depth, and
the structured hierarchy section is the only hierarchy route the context offers."""

from __future__ import annotations

from pathlib import Path

import pytest

from showcase.ontology_query_benchmark.sema_context import ANCHOR_DEPTH_CEILING, anchor_bounds
from tests.unit.ontology_query_benchmark import sema_hierarchy_acris_env as acris
from tests.unit.ontology_query_benchmark.hierarchy_fixture import hierarchy, warehouse
from tests.unit.ontology_query_benchmark.sema_hierarchy_env import _build, _facts, _reader
from tests.unit.ontology_query_benchmark.sema_hierarchy_anchor_env import (
    EVENTS_SQL,
    READABLE,
    anchor,
    context,
    deeper_facts,
    deeper_warehouse,
    fragments,
    members,
)

pytestmark = pytest.mark.unit


def test_a_retrieved_anchor_fills_the_fragment_and_is_named(tmp_path: Path) -> None:
    text = context(tmp_path, _facts(tmp_path, hierarchy()), READABLE,
                   [anchor("C1", "Neoplasm", "std", 2)]).text  # fmt: skip

    section = text[text.index("CONCEPT HIERARCHY"):]
    assert "anchor name 'Neoplasm', vocabulary 'std'; 2 members" in section
    [fragment] = fragments(text, "site_S1")
    assert members(warehouse(), EVENTS_SQL, fragment) == {"L10", "L11"}


def test_each_anchor_gets_its_own_filled_fragment(tmp_path: Path) -> None:
    text = context(tmp_path, _facts(tmp_path, hierarchy()), READABLE, [
        anchor("C1", "Neoplasm", "std", 2), anchor("C3", "Small cell lung neoplasm", "std", 1),
    ]).text  # fmt: skip

    found = [members(warehouse(), EVENTS_SQL, f) for f in fragments(text, "site_S1")]
    assert found == [{"L10", "L11"}, {"L10"}]


def test_an_anchor_on_another_column_leaves_the_section_as_before(tmp_path: Path) -> None:
    facts = _facts(tmp_path, hierarchy())
    plain = context(tmp_path, facts, READABLE, []).text
    other = context(tmp_path, facts, READABLE, [anchor("C1", "Neoplasm", "std", 2, "dx")]).text

    assert other == plain
    assert fragments(plain, "site_S1", filled=False) and not fragments(plain, "site_S1")


def test_an_acris_class_anchor_resolves_its_document_types(tmp_path: Path) -> None:
    con = acris._acris_warehouse()
    text = context(tmp_path, acris._facts(tmp_path, con), dict.fromkeys(acris.RELATIONS, True),
                   [anchor(acris.CLASS, acris.CLASS, "lib", 23)]).text  # fmt: skip

    [fragment] = fragments(text, "site_S1")
    expected = {r["DOC. TYPE"] for r in acris.ROWS if r["CLASS CODE DESCRIPTION"] == acris.CLASS}
    source = "SELECT doc_type AS site_S1 FROM wh.acris.master"
    assert members(con, source, fragment) == expected and len(expected) == 23


def test_an_acris_class_without_a_concept_row_fills_a_predicate_that_resolves(
    tmp_path: Path,
) -> None:
    con = acris._acris_warehouse(class_rows=False)
    text = context(tmp_path, acris._facts(tmp_path, con), dict.fromkeys(acris.RELATIONS, True),
                   [anchor(acris.CLASS, acris.CLASS, "lib", 23)]).text  # fmt: skip

    lines = [x.strip() for x in text.splitlines()]
    filled = lines[next(i for i, x in enumerate(lines) if x.startswith("Filled for anchor")) + 1]
    expected = {r["DOC. TYPE"] for r in acris.ROWS if r["CLASS CODE DESCRIPTION"] == acris.CLASS}
    source = "SELECT doc_type AS site_S1 FROM wh.acris.master"
    assert members(con, source, filled.removeprefix("WHERE ")) == expected


def test_the_structured_section_replaces_the_prose_hierarchy_reference(tmp_path: Path) -> None:
    text = context(tmp_path, _facts(tmp_path, hierarchy()), READABLE, []).text

    assert "CONCEPT HIERARCHY" in text
    assert "TOY HIERARCHY REFERENCE" not in text


def test_without_the_structured_section_the_prose_reference_stays(tmp_path: Path) -> None:
    blocked = {**READABLE, "wh.voc.ancestor": False}
    text = context(tmp_path, _facts(tmp_path, hierarchy()), blocked, []).text

    assert "CONCEPT HIERARCHY" not in text
    assert "TOY HIERARCHY REFERENCE" in text


def test_the_build_records_the_slice_depth(tmp_path: Path) -> None:
    _, shallow = _build(tmp_path / "a", _facts(tmp_path / "a", hierarchy()), _reader())
    _, deep = _build(tmp_path / "b", deeper_facts(tmp_path / "b"), _reader())

    assert shallow.counts["hierarchy_depth"] == 2
    assert deep.counts["hierarchy_depth"] == 4


def test_retrieval_walks_as_deep_as_the_slice_within_the_ceiling(tmp_path: Path) -> None:
    _, deep = _build(tmp_path, deeper_facts(tmp_path), _reader())
    _, plain = _build(tmp_path / "p", _facts(tmp_path / "p"), None)
    over = type(deep)(**{**deep.__dict__,
                         "counts": {**deep.counts, "hierarchy_depth": ANCHOR_DEPTH_CEILING + 5}})

    bounds = anchor_bounds(deep)
    assert bounds is not None and bounds.max_depth == 4
    assert anchor_bounds(plain) is None
    capped = anchor_bounds(over)
    assert capped is not None and capped.max_depth == ANCHOR_DEPTH_CEILING


def test_an_anchor_four_levels_up_fills_an_executable_fragment(tmp_path: Path) -> None:
    text = context(tmp_path, deeper_facts(tmp_path), READABLE,
                   [anchor("R", "Clinical finding", "std", 2)]).text  # fmt: skip

    [fragment] = fragments(text, "site_S1")
    assert members(deeper_warehouse(), EVENTS_SQL, fragment) == {"L10", "L11"}


@pytest.mark.parametrize(("denied", "relation", "column"), [
    ("ANC", "wh.voc.ancestor", "anc"), ("flag", "wh.voc.concept", "flag"),
    ("rel", "wh.voc.relationship", "rel"), ("cname", "wh.voc.concept", "cname"),
])  # fmt: skip
def test_a_hierarchy_naming_a_denied_column_is_withheld_with_a_finding(
    tmp_path: Path, denied: str, relation: str, column: str
) -> None:
    ctx = context(tmp_path, _facts(tmp_path, hierarchy()), READABLE,
                  [anchor("C1", "Neoplasm", "std", 2)], lambda n: n.lower() == denied.lower())

    assert "CONCEPT HIERARCHY" not in ctx.text and "Neoplasm" not in ctx.text
    assert "TOY HIERARCHY REFERENCE" not in ctx.text
    withheld = [(f.relation, f.column) for f in ctx.findings if "concept hierarchy" in f.message]
    assert withheld == [(relation, column)]


def test_a_policy_denying_no_hierarchy_column_leaves_the_section_as_before(
    tmp_path: Path,
) -> None:
    facts, anchors = _facts(tmp_path, hierarchy()), [anchor("C1", "Neoplasm", "std", 2)]
    plain = context(tmp_path, facts, READABLE, anchors).text
    policed = context(tmp_path, facts, READABLE, anchors, lambda n: n == "lvl")

    assert policed.text == plain
    assert "concept hierarchy" not in " ".join(f.message for f in policed.findings)


def test_an_anchor_on_a_denied_column_renders_no_anchor(tmp_path: Path) -> None:
    text = context(tmp_path, _facts(tmp_path, hierarchy()), READABLE,
                   [anchor("C1", "Neoplasm", "std", 2)], lambda n: n == "site_S1").text  # fmt: skip

    assert "Neoplasm" not in text and "site_S1" not in text
