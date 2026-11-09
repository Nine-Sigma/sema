"""A concept hierarchy reaches arm S as graph edges, value bridges and executable SQL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from showcase.ontology_query_benchmark.profile import Attribution, ConceptHierarchy
from showcase.ontology_query_benchmark.sema_bindings import (
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.ontology_query_benchmark.sema_build import (
    BuildManifest,
    build_scoped_graph,
    read_manifest,
    write_manifest,
)
from showcase.ontology_query_benchmark.sema_build_utils import LibraryFacts, import_library_facts
from showcase.ontology_query_benchmark.sema_context import SContext, SContextProvider
from showcase.ontology_query_benchmark.sema_hierarchy_sql import membership_sql
from tests.unit.ontology_query_benchmark.hierarchy_fixture import (
    ANCHOR,
    MEMBERS,
    add_hierarchy_slice,
    add_typed_vocabulary,
    hierarchy,
    typed_hierarchy,
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
from tests.unit.ontology_query_benchmark.test_sema_build import _catalog, _embedder
from tests.unit.ontology_query_benchmark.test_sema_context import (
    EVENTS_ENTITY,
    LIMITS,
    _Retriever,
)

pytestmark = pytest.mark.unit

RELATIONS = ("wh.voc.relationship", "wh.voc.ancestor", "wh.voc.concept")
SITE = ("Site of event", "Site code.", [(m, f"site {m}") for m in MEMBERS])


def _facts(tmp_path: Path, spec: ConceptHierarchy | None = None) -> LibraryFacts:
    projection, receipt = write_projection(tmp_path, {**ELEMENTS, "S1": SITE})
    add_hierarchy_slice(projection, receipt)
    return import_library_facts(CONCEPT_SPEC, projection, receipt,
                                spec_attribution=Attribution.ASSISTED, hierarchy=spec)  # fmt: skip


def _reader(unreadable: str | None = None) -> MagicMock:
    def read(sql: str) -> list[Any]:
        if unreadable and unreadable in sql:
            raise RuntimeError(f"TABLE_OR_VIEW_NOT_FOUND {unreadable}")
        return [(1,)]

    return MagicMock(side_effect=read)


def _build(
    tmp_path: Path, facts: LibraryFacts, reader: Any = None
) -> tuple[MagicMock, BuildManifest]:
    loader = MagicMock()
    manifest = build_scoped_graph(
        loader, _catalog(), load_sidecar(write_fixture(tmp_path), tmp_path), facts, _embedder(),
        dimensions=4, embedder_model="toy-embed", relation_reader=reader,
    )
    return loader, manifest


def _bridges(loader: MagicMock) -> set[tuple[str, str, str, str]]:
    return {(k["sv"], k["sc"], k["cv"], k["cc"]) for c in loader._run.call_args_list
            if "MAPS_TO_CONCEPT" in c.args[0] for k in [c.kwargs]}  # fmt: skip


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


def _context(tmp_path: Path, facts: LibraryFacts, readable: dict[str, bool] | None) -> SContext:
    sidecar = load_sidecar(write_fixture(tmp_path), tmp_path)
    products = load_products(sidecar)
    provider = SContextProvider(
        _Retriever([EVENTS_ENTITY]), sidecar, products,
        resolve_bindings(sidecar, products, catalog_fixture()), facts,
        graph_identity="graph-1", conventions=CONVENTIONS, limits=LIMITS,
        hierarchy_relations=readable,
    )
    return provider("Which events sit under a broader concept?")


def _fragment(text: str, column: str) -> str:
    line = next(x for x in text.splitlines() if x.strip().startswith(f"WHERE {column} IN"))
    return line.strip().removeprefix("WHERE ")


def test_the_fragment_returns_the_descendant_members_including_the_replaced_one(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, _facts(tmp_path, hierarchy()), dict.fromkeys(RELATIONS, True))

    fragment = _fragment(context.text, "site_S1").replace("{anchor_id}", f"'{ANCHOR}'")
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
    sql = membership_sql(typed_hierarchy(), "dx").replace("{anchor_id}", "1")

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
