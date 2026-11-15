"""A concept hierarchy reaches arm S as graph edges, value bridges and executable SQL."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import duckdb
import pytest

from showcase.ontology_query_benchmark.profile import Attribution, ConceptHierarchy
from showcase.ontology_query_benchmark.sema_build import read_manifest, write_manifest
from showcase.ontology_query_benchmark.sema_build_utils import import_library_facts
from showcase.ontology_query_benchmark.sema_hierarchy_sql import membership_sql
from tests.unit.ontology_query_benchmark.hierarchy_fixture import (
    add_typed_vocabulary,
    closure_rows,
    hierarchy,
    table,
    typed_hierarchy,
    warehouse,
)
from tests.unit.ontology_query_benchmark.sema_bindings_env import (
    CONCEPT_SPEC,
    ELEMENTS,
    EVENTS,
    write_projection,
)
from tests.unit.ontology_query_benchmark.sema_hierarchy_env import (
    RELATIONS,
    SITE,
    _bridges,
    _build,
    _context,
    _facts,
    _fragment,
    _reader,
    require_name_keyed_template,
)

pytestmark = pytest.mark.unit


def _edges(loader: MagicMock) -> set[tuple[str, str, str]]:
    return {(c.kwargs["parent_code"], c.kwargs["child_code"], c.kwargs["vocabulary_name"])
            for c in loader.add_term_hierarchy.call_args_list}  # fmt: skip


def test_the_build_writes_parent_edges_and_standard_bridges(tmp_path: Path) -> None:
    loader, manifest = _build(tmp_path, _facts(tmp_path, hierarchy()), _reader())

    assert _edges(loader) == {
        ("C1", "C2", "std"), ("C2", "C3", "std"), ("C2", "C5", "std"), ("C2", "C7", "std"),
    }
    assert _bridges(loader) == {
        ("lib", "L10", "std", "C3"), ("lib", "L11", "std", "C5"), ("lib", "L12", "std", "C6"),
    }
    labels = {c.args[0]: c.args[1] for c in loader.upsert_term.call_args_list
              if c.kwargs.get("vocabulary_name") == "std"}  # fmt: skip
    assert labels["C1"] == "Neoplasm"
    assert manifest.counts["value_bridges"] == 3
    write_manifest(manifest, tmp_path / "manifest.json")
    assert read_manifest(tmp_path / "manifest.json").hierarchy_relations == dict.fromkeys(
        RELATIONS, True
    )


def test_each_relation_is_read_once_with_limit_one(tmp_path: Path) -> None:
    reader = _reader()
    _build(tmp_path, _facts(tmp_path, hierarchy()), reader)

    assert sorted(c.args[0] for c in reader.call_args_list) == sorted(
        f"SELECT 1 FROM {r} LIMIT 1" for r in RELATIONS
    )


def test_an_unreadable_relation_is_a_build_finding(tmp_path: Path) -> None:
    _, manifest = _build(tmp_path, _facts(tmp_path, hierarchy()), _reader("wh.voc.ancestor"))

    assert manifest.hierarchy_relations is not None
    assert manifest.hierarchy_relations["wh.voc.ancestor"] is False
    assert any("insufficient evidence" in f and "wh.voc.ancestor" in f for f in manifest.findings)


def test_without_a_hierarchy_the_build_writes_no_edge_bridge_or_manifest_key(
    tmp_path: Path,
) -> None:
    loader, manifest = _build(tmp_path, _facts(tmp_path))

    assert not _edges(loader) and not _bridges(loader)
    write_manifest(manifest, tmp_path / "manifest.json")
    assert "hierarchy_relations" not in json.loads((tmp_path / "manifest.json").read_text())


def test_the_fragment_returns_the_descendant_members_including_the_replaced_one(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, _facts(tmp_path, hierarchy()), dict.fromkeys(RELATIONS, True))

    fragment = _fragment(context.text, "site_S1").replace("{anchor_name}", "'Neoplasm'")
    found = warehouse().execute(
        f"SELECT DISTINCT site_S1 FROM (SELECT dx AS site_S1 FROM wh.data.events) WHERE {fragment}"
    ).fetchall()
    assert {row[0] for row in found} == {"L10", "L11"}


def test_the_fragment_states_identifiers_direction_and_self_policy(tmp_path: Path) -> None:
    context = _context(tmp_path, _facts(tmp_path, hierarchy()), dict.fromkeys(RELATIONS, True))

    section = context.text[context.text.index("CONCEPT HIERARCHY"):]
    assert f"{EVENTS}.site_S1 holds source member ids" in section
    assert "anc is the broader concept" in section
    assert "self rows are included" in section
    assert "'Replaced by'" in section


def test_an_unreadable_relation_withholds_the_fragment(tmp_path: Path) -> None:
    readable = {**dict.fromkeys(RELATIONS, True), "wh.voc.ancestor": False}
    context = _context(tmp_path, _facts(tmp_path, hierarchy()), readable)

    assert "WHERE site_S1 IN" not in context.text
    finding = next(f for f in context.findings if "insufficient evidence" in f.message)
    assert finding.column == "site_S1" and "wh.voc.ancestor" in finding.message


def test_without_a_hierarchy_the_context_has_no_hierarchy_section(tmp_path: Path) -> None:
    context = _context(tmp_path, _facts(tmp_path), None)

    assert "CONCEPT HIERARCHY" not in context.text
    assert not any("insufficient evidence" in f.message for f in context.findings)


def test_a_declared_target_type_casts_text_mapping_targets() -> None:
    sql = membership_sql(typed_hierarchy(), "dx", dialect="duckdb", name="Neoplasm")

    found = add_typed_vocabulary(warehouse()).execute(
        f"SELECT dx FROM (VALUES ('L10'), ('L11'), ('L12')) v(dx) WHERE {sql}"
    ).fetchall()
    assert found == [("L10",)]


def test_a_projection_without_the_slice_is_refused_with_the_remedy(tmp_path: Path) -> None:
    projection, receipt = write_projection(tmp_path, {**ELEMENTS, "S1": SITE})

    with pytest.raises(ValueError, match="rerun the extract"):
        import_library_facts(
            CONCEPT_SPEC, projection, receipt,
            spec_attribution=Attribution.ASSISTED, hierarchy=hierarchy(),
        )


LOAN_SPEC: dict[str, Any] = {
    "attribution": "assisted",
    "closure": {"relation": "wh.kb.closure", "ancestor_column": "anc",
                "descendant_column": "des", "level_column": "lvl", "self_rows": True},
    "concepts": {"relation": "wh.kb.concept", "id_column": "cid", "name_column": "cname",
                 "vocabulary_column": "vocab", "class_column": "kind",
                 "standard_column": "flag", "standard_values": ["S"]},
}  # fmt: skip
LOAN_CONCEPTS = [
    ("K1", "Bank loan", "alpha", "Product", "S"), ("K2", "Bank loan", "beta", "Product", "S"),
    ("K3", "Bank loan", "alpha", "Product", None), ("K7", "Bank loan", "alpha", "Channel", "S"),
    ("K4", "Car loan", "alpha", "Product", "S"), ("K5", "Home loan", "alpha", "Product", "S"),
    ("K6", "Student loan", "beta", "Product", "S"), ("K8", "Payday loan", "alpha", "Product", "S"),
    ("K10", "Branch", "alpha", "Channel", "S"),
]
LOAN_EDGES = [("K1", "K4"), ("K1", "K5"), ("K2", "K6"), ("K3", "K8"), ("K7", "K10")]


def _loan_warehouse() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("ATTACH ':memory:' AS wh")
    con.execute("CREATE SCHEMA wh.kb")
    table(con, "wh.kb.concept", "cid VARCHAR, cname VARCHAR, vocab VARCHAR, kind VARCHAR, "
          "flag VARCHAR", LOAN_CONCEPTS)  # fmt: skip
    table(con, "wh.kb.closure", "anc VARCHAR, des VARCHAR, lvl INTEGER",
          sorted(closure_rows(LOAN_EDGES, self_rows=True)))  # fmt: skip
    return con


@pytest.mark.parametrize("own_ids", [False, True])
def test_the_template_keys_the_anchor_by_name_not_by_a_listed_id(own_ids: bool) -> None:
    template = membership_sql(hierarchy(), "{column}", dialect="duckdb", own_ids=own_ids)

    assert "{anchor_id}" not in template
    assert re.search(r"\bcname = ", template), template


def test_a_qualified_name_renders_name_and_qualifier_filters_without_an_id() -> None:
    require_name_keyed_template()
    spec = ConceptHierarchy.model_validate(LOAN_SPEC)

    sql = membership_sql(spec, "dx", dialect="duckdb", name="Bank loan", vocabulary="alpha",
                         concept_class="Product")  # fmt: skip

    for predicate in ("cname = 'Bank loan'", "vocab = 'alpha'", "kind = 'Product'", "flag IN"):
        assert predicate in sql, sql
    assert not {cid for cid, *_ in LOAN_CONCEPTS} & set(re.findall(r"'([^']*)'", sql))


def test_a_shared_name_selects_only_the_qualified_standard_concept_members() -> None:
    require_name_keyed_template()
    spec = ConceptHierarchy.model_validate(LOAN_SPEC)
    sql = membership_sql(spec, "dx", dialect="duckdb", name="Bank loan", vocabulary="alpha",
                         concept_class="Product")  # fmt: skip

    found = _loan_warehouse().execute(
        "SELECT dx FROM (VALUES ('K4'), ('K5'), ('K6'), ('K8'), ('K10')) v(dx)"
        f" WHERE {sql} ORDER BY dx"
    ).fetchall()
    assert [row[0] for row in found] == ["K4", "K5"]
