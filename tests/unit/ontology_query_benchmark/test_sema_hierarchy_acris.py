"""A non-library instance: document types under their class in a public code table, a
parent-column hierarchy whose closure answers "documents that are a kind of X"."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pytest

from showcase.ontology_query_benchmark.profile import ConceptHierarchy
from showcase.ontology_query_benchmark.sema_hierarchy_sql import membership_sql
from tests.unit.ontology_query_benchmark.hierarchy_fixture import closure_rows, table
from tests.unit.ontology_query_benchmark.sema_hierarchy_env import (
    _build,
    _context,
    _fragment,
    require_name_keyed_template,
)
from tests.unit.ontology_query_benchmark.sema_hierarchy_acris_env import (
    CLASS,
    RELATIONS,
    ROWS,
    SPEC,
    _acris_warehouse,
    _facts,
)

pytestmark = pytest.mark.unit


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


def test_a_class_without_a_concept_row_resolves_through_the_closure(tmp_path: Path) -> None:
    con = _acris_warehouse(class_rows=False)
    context = _context(tmp_path, _facts(tmp_path, con), dict.fromkeys(RELATIONS, True))

    keyed = [x.strip().removeprefix("WHERE ") for x in context.text.splitlines()
             if "{column} IN" in x and "{anchor_id}" in x]  # fmt: skip
    assert keyed, "no template keys an ancestor that has no concept row by its id"
    fragment = keyed[0].replace("{column}", "site_S1").replace("{anchor_id}", f"'{CLASS}'")
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
