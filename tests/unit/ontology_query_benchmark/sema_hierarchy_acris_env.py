"""Shared setup for test_sema_hierarchy_acris.py and the tests that reuse it."""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb

from showcase.ontology_query_benchmark.intake_extract import extract
from showcase.ontology_query_benchmark.profile import Attribution, ConceptHierarchy
from showcase.ontology_query_benchmark.sema_build_utils import LibraryFacts, import_library_facts
from tests.unit.ontology_query_benchmark.hierarchy_fixture import (
    WarehouseCursor,
    closure_rows,
    copy_hierarchy_slice,
    spec_with,
    table,
    warehouse,
    write_product_spec,
)
from tests.unit.ontology_query_benchmark.sema_bindings_env import (
    CONCEPT_SPEC,
    ELEMENTS,
    write_projection,
)

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


def _acris_warehouse(*, class_rows: bool = True) -> duckdb.DuckDBPyConnection:
    """Without ``class_rows`` a class is only a closure ancestor, as in the code table."""
    con = warehouse(tuple(r["DOC. TYPE"] for r in ROWS))
    con.execute("CREATE SCHEMA wh.acris")
    edges = [(r["CLASS CODE DESCRIPTION"], r["DOC. TYPE"]) for r in ROWS]
    table(con, "wh.acris.closure", "ancestor VARCHAR, descendant VARCHAR, levels INTEGER",
          sorted(closure_rows(edges, self_rows=True)))  # fmt: skip
    classes = {(c, c) for c in {r["CLASS CODE DESCRIPTION"] for r in ROWS}} if class_rows else ()
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
