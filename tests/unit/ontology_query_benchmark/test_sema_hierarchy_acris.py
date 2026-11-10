"""A non-library instance: document types under their class in a public code table, a
parent-column hierarchy whose closure answers "documents that are a kind of X"."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pytest

from showcase.ontology_query_benchmark.intake_extract import extract
from showcase.ontology_query_benchmark.profile import Attribution, ConceptHierarchy
from showcase.ontology_query_benchmark.sema_build_utils import LibraryFacts, import_library_facts
from showcase.ontology_query_benchmark.sema_hierarchy_sql import membership_sql
from tests.unit.ontology_query_benchmark.hierarchy_fixture import (
    WarehouseCursor,
    closure_rows,
    copy_hierarchy_slice,
    spec_with,
    table,
    warehouse,
    write_product_spec,
)
from tests.unit.ontology_query_benchmark.test_sema_bindings import (
    CONCEPT_SPEC,
    ELEMENTS,
    write_projection,
)
from tests.unit.ontology_query_benchmark.test_sema_hierarchy import (
    _build,
    _context,
    _fragment,
    require_name_keyed_template,
)

pytestmark = pytest.mark.unit

CODES = Path(__file__).parents[2] / "fixtures/intake/acris/Document_Control_Codes.csv"
ROWS = list(csv.DictReader(CODES.open(encoding="utf-8")))
CLASS = "MORTGAGES & INSTRUMENTS"
RELATIONS = ("wh.acris.closure", "wh.acris.concept")
SPEC = ConceptHierarchy.model_validate({
    "attribution": "assisted",
    "closure": {"relation": "wh.acris.closure", "ancestor_column": "ancestor",
                "descendant_column": "descendant", "level_column": "levels", "self_rows": True},
    "concepts": {"relation": "wh.acris.concept", "id_column": "doc_type",
                 "name_column": "description"},
})  # fmt: skip


def _acris_warehouse() -> duckdb.DuckDBPyConnection:
    con = warehouse(tuple(r["DOC. TYPE"] for r in ROWS))
    con.execute("CREATE SCHEMA wh.acris")
    edges = [(r["CLASS CODE DESCRIPTION"], r["DOC. TYPE"]) for r in ROWS]
    table(con, "wh.acris.closure", "ancestor VARCHAR, descendant VARCHAR, levels INTEGER",
          sorted(closure_rows(edges, self_rows=True)))  # fmt: skip
    classes = {(c, c) for c in {r["CLASS CODE DESCRIPTION"] for r in ROWS}}
    table(con, "wh.acris.concept", "doc_type VARCHAR, description VARCHAR",
          [(r["DOC. TYPE"], r["DOC. TYPE DESCRIPTION"]) for r in ROWS] + sorted(classes))
    table(con, "wh.acris.master", "document_id INTEGER, doc_type VARCHAR",
          [(i, r["DOC. TYPE"]) for i, r in enumerate(ROWS * 2)])  # fmt: skip
    return con


def _facts(tmp_path: Path, con: duckdb.DuckDBPyConnection) -> LibraryFacts:
    receipt = extract(lambda: WarehouseCursor(con), spec_with(SPEC),
                      [write_product_spec(tmp_path / "events.json")], tmp_path / "out")  # fmt: skip
    assert receipt["complete"] is True, receipt["incomplete"]
    members = [(r["DOC. TYPE"], r["DOC. TYPE DESCRIPTION"]) for r in ROWS]
    projection, pinned = write_projection(
        tmp_path / "lib", {**ELEMENTS, "S1": ("Document type", "Document type code.", members)}
    )
    copy_hierarchy_slice(tmp_path / "out" / receipt["duckdb"]["path"], projection, pinned)
    return import_library_facts(CONCEPT_SPEC, projection, pinned,
                                spec_attribution=Attribution.ASSISTED, hierarchy=SPEC)  # fmt: skip


def test_the_class_parent_column_becomes_parent_edges_in_the_member_vocabulary(
    tmp_path: Path,
) -> None:
    loader, manifest = _build(tmp_path, _facts(tmp_path, _acris_warehouse()), MagicMock())

    edges = {(c.kwargs["parent_code"], c.kwargs["child_code"], c.kwargs["vocabulary_name"])
             for c in loader.add_term_hierarchy.call_args_list}  # fmt: skip
    assert len(edges) == len(ROWS)
    assert (CLASS, "SAT", "lib") in edges
    assert not [c for c in loader._run.call_args_list if "MAPS_TO_CONCEPT" in c.args[0]]
    labels = {c.args[0]: c.args[1] for c in loader.upsert_term.call_args_list}
    assert labels[CLASS] == CLASS
    assert manifest.hierarchy_relations == dict.fromkeys(RELATIONS, True)


def test_documents_of_a_kind_resolve_through_the_closure(tmp_path: Path) -> None:
    con = _acris_warehouse()
    context = _context(tmp_path, _facts(tmp_path, con), dict.fromkeys(RELATIONS, True))

    assert "site_S1 holds concept ids of the hierarchy" in context.text
    fragment = _fragment(context.text, "site_S1").replace("{anchor_name}", f"'{CLASS}'")
    found = con.execute(
        "SELECT DISTINCT site_S1 FROM (SELECT doc_type AS site_S1 FROM wh.acris.master)"
        f" WHERE {fragment}"
    ).fetchall()
    expected = {r["DOC. TYPE"] for r in ROWS if r["CLASS CODE DESCRIPTION"] == CLASS}
    assert {row[0] for row in found} == expected and len(expected) == 23


KIND_CONCEPTS = [
    ("class-1", CLASS, "document control", "Document class"),
    ("class-2", "DEEDS AND OTHER CONVEYANCES", "document control", "Document class"),
    ("SAT", "SATISFACTION OF MORTGAGE", "document control", "Document type"),
    ("ASPM", "ASSUMPTION OF MORTGAGE", "document control", "Document type"),
    ("SUBM", "SUBORDINATION OF MORTGAGE", "document control", "Document type"),
    ("CORRD", "CORRECTION DEED", "document control", "Document type"),
]
KIND_EDGES = [("class-1", "SAT"), ("class-1", "ASPM"), ("class-1", "SUBM"), ("class-2", "CORRD")]


def test_document_types_of_a_kind_are_selected_by_class_name_not_id() -> None:
    require_name_keyed_template()
    con = duckdb.connect()
    con.execute("ATTACH ':memory:' AS wh")
    con.execute("CREATE SCHEMA wh.acris")
    table(con, "wh.acris.concept",
          "doc_type VARCHAR, description VARCHAR, code_list VARCHAR, level VARCHAR",
          KIND_CONCEPTS)  # fmt: skip
    table(con, "wh.acris.closure", "ancestor VARCHAR, descendant VARCHAR, levels INTEGER",
          sorted(closure_rows(KIND_EDGES, self_rows=True)))  # fmt: skip

    spec = ConceptHierarchy.model_validate({
        **SPEC.model_dump(mode="json", exclude_none=True),
        "concepts": {"relation": "wh.acris.concept", "id_column": "doc_type",
                     "name_column": "description", "vocabulary_column": "code_list",
                     "class_column": "level"},
    })  # fmt: skip
    sql = membership_sql(spec, "doc_type", dialect="duckdb", name=CLASS,
                         concept_class="Document class")  # fmt: skip

    assert "class-1" not in sql
    found = con.execute(
        "SELECT doc_type FROM (VALUES ('SAT'), ('ASPM'), ('SUBM'), ('CORRD')) v(doc_type)"
        f" WHERE {sql}"
    ).fetchall()
    assert {row[0] for row in found} == {"SAT", "ASPM", "SUBM"}
