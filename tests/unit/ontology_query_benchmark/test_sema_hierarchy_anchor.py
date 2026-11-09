"""A retrieved hierarchy anchor fills arm S's fragment, the slice sets the walk depth, and
the structured hierarchy section is the only hierarchy route the context offers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import pytest

from showcase.ontology_query_benchmark.profile import Attribution
from showcase.ontology_query_benchmark.sema_bindings import (
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.ontology_query_benchmark.sema_build_utils import LibraryFacts, import_library_facts
from showcase.ontology_query_benchmark.sema_context import (
    ANCHOR_DEPTH_CEILING,
    SContext,
    SContextProvider,
    anchor_bounds,
)
from showcase.ontology_query_benchmark.sema_context_utils import nothing_denied
from tests.unit.ontology_query_benchmark import test_sema_hierarchy_acris as acris
from tests.unit.ontology_query_benchmark.hierarchy_fixture import (
    add_hierarchy_slice,
    hierarchy,
    repin,
    table,
    warehouse,
)
from tests.unit.ontology_query_benchmark.test_sema_bindings import (
    CONCEPT_SPEC,
    CONVENTIONS,
    ELEMENTS,
    EVENTS,
    catalog_fixture,
    write_fixture,
    write_projection,
)
from tests.unit.ontology_query_benchmark.test_sema_context import (
    EVENTS_ENTITY,
    LIMITS,
    _Retriever,
)
from tests.unit.ontology_query_benchmark.test_sema_hierarchy import (
    RELATIONS,
    SITE,
    _build,
    _facts,
    _reader,
)

pytestmark = pytest.mark.unit

READABLE = dict.fromkeys(RELATIONS, True)
DEEPER_CONCEPTS = [("C0", "Disorder", "S"), ("R", "Clinical finding", "S")]
DEEPER_EDGES = [("C0", "C1"), ("R", "C0")]


def anchor(code: str, label: str, vocabulary: str, count: int,
           column: str = "site_S1") -> dict[str, Any]:  # fmt: skip
    return {"type": "hierarchy_anchor", "code": code, "label": label,
            "vocabulary": vocabulary, "table": EVENTS, "column": column, "schema": "sch",
            "member_count": count, "member_codes": [], "status": "auto",
            "confidence": 1.0, "confidence_policy": "semantic"}  # fmt: skip


def context(tmp_path: Path, facts: LibraryFacts, readable: dict[str, bool] | None,
            anchors: list[dict[str, Any]],
            is_denied: Callable[[str], bool] = nothing_denied) -> SContext:  # fmt: skip
    sidecar = load_sidecar(write_fixture(tmp_path), tmp_path)
    products = load_products(sidecar)
    provider = SContextProvider(
        _Retriever([EVENTS_ENTITY, *anchors]), sidecar, products,
        resolve_bindings(sidecar, products, catalog_fixture()), facts,
        graph_identity="graph-1", conventions=CONVENTIONS, limits=LIMITS,
        hierarchy_relations=readable,
    )
    return provider("Which events sit under a broader concept?", is_denied)


def fragments(text: str, column: str, *, filled: bool = True) -> list[str]:
    """The column's anchor-filled fragments, or with ``filled=False`` its template."""
    lines = [x.strip().removeprefix("WHERE ") for x in text.splitlines()]
    found = [x for x in lines if x.lstrip("(").startswith(f"{column} IN")]
    return [f for f in found if ("{anchor_id}" in f) != filled]


def members(con: duckdb.DuckDBPyConnection, source: str, fragment: str) -> set[str]:
    found = con.execute(f"SELECT DISTINCT site_S1 FROM ({source}) WHERE {fragment}")
    return {row[0] for row in found.fetchall()}


EVENTS_SQL = "SELECT dx AS site_S1 FROM wh.data.events"


def test_a_retrieved_anchor_fills_the_fragment_and_is_named(tmp_path: Path) -> None:
    text = context(tmp_path, _facts(tmp_path, hierarchy()), READABLE,
                   [anchor("C1", "Neoplasm", "std", 2)]).text  # fmt: skip

    section = text[text.index("CONCEPT HIERARCHY"):]
    assert "Neoplasm" in section and "C1" in section and "std" in section
    assert "2 members" in section
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


def test_the_structured_section_replaces_the_prose_hierarchy_reference(tmp_path: Path) -> None:
    text = context(tmp_path, _facts(tmp_path, hierarchy()), READABLE, []).text

    assert "CONCEPT HIERARCHY" in text
    assert "TOY HIERARCHY REFERENCE" not in text


def test_without_the_structured_section_the_prose_reference_stays(tmp_path: Path) -> None:
    blocked = {**READABLE, "wh.voc.ancestor": False}
    text = context(tmp_path, _facts(tmp_path, hierarchy()), blocked, []).text

    assert "CONCEPT HIERARCHY" not in text
    assert "TOY HIERARCHY REFERENCE" in text


def deeper_facts(tmp_path: Path) -> LibraryFacts:
    """The synthetic slice with two more levels above its top concept: depth 4."""
    projection, receipt = write_projection(tmp_path, {**ELEMENTS, "S1": SITE})
    add_hierarchy_slice(projection, receipt)
    with duckdb.connect(str(projection)) as con:
        con.executemany("INSERT INTO concept_hierarchy.concept VALUES (?, ?, ?)", DEEPER_CONCEPTS)
        con.executemany("INSERT INTO concept_hierarchy.edge VALUES (?, ?)", DEEPER_EDGES)
    repin(projection, receipt)
    return import_library_facts(
        CONCEPT_SPEC, projection, receipt,
        spec_attribution=Attribution.ASSISTED, hierarchy=hierarchy(),
    )


def deeper_warehouse() -> duckdb.DuckDBPyConnection:
    con = warehouse()
    table(con, "wh.voc.extra", "cid VARCHAR, cname VARCHAR, flag VARCHAR", DEEPER_CONCEPTS)
    con.execute("INSERT INTO wh.voc.concept SELECT * FROM wh.voc.extra")
    below_c1 = [r[0] for r in con.execute(
        "SELECT des FROM wh.voc.ancestor WHERE anc = 'C1'").fetchall()]  # fmt: skip
    rows = [("C0", "C0", 0), ("R", "R", 0), ("R", "C0", 1)]
    rows += [(a, d, 0) for a in ("C0", "R") for d in below_c1]
    con.executemany("INSERT INTO wh.voc.ancestor VALUES (?, ?, ?)", rows)
    return con


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
    ("rel", "wh.voc.relationship", "rel"),
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
    policed = context(tmp_path, facts, READABLE, anchors, lambda n: n in ("cname", "lvl"))

    assert policed.text == plain
    assert "concept hierarchy" not in " ".join(f.message for f in policed.findings)


def test_an_anchor_on_a_denied_column_renders_no_anchor(tmp_path: Path) -> None:
    text = context(tmp_path, _facts(tmp_path, hierarchy()), READABLE,
                   [anchor("C1", "Neoplasm", "std", 2)], lambda n: n == "site_S1").text  # fmt: skip

    assert "Neoplasm" not in text and "site_S1" not in text
