"""Scoped retrieval-graph build for arm S (Neo4j and embedder mocked)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.graph.physical_catalog import PhysicalColumn, StaticPhysicalCatalogSource
from showcase.ontology_query_benchmark.sema_bindings import SpecPinError, load_sidecar
from showcase.ontology_query_benchmark.sema_build import (
    BuildManifest,
    build_scoped_graph,
    read_manifest,
    write_manifest,
)
from showcase.ontology_query_benchmark.sema_build_utils import (
    FactsConceptSource,
    LibraryFacts,
    import_library_facts,
)
from tests.unit.ontology_query_benchmark.test_sema_bindings import (
    CONCEPT_SPEC,
    PEOPLE,
    ELEMENTS,
    EVENTS,
    catalog_fixture,
    facts_fixture,
    write_fixture,
    write_projection,
)

pytestmark = pytest.mark.unit


def _catalog() -> StaticPhysicalCatalogSource:
    return StaticPhysicalCatalogSource({
        ("cat", "sch", table): [PhysicalColumn(c, t, True) for c, t in cols.items()]
        for table, cols in catalog_fixture().items()
    })


def _embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_documents.side_effect = lambda texts: [[0.5] * 4 for _ in texts]
    return embedder


def _build_with(tmp_path: Path, facts: LibraryFacts) -> tuple[MagicMock, MagicMock, BuildManifest]:
    loader, embedder = MagicMock(), _embedder()
    manifest = build_scoped_graph(
        loader, _catalog(), load_sidecar(write_fixture(tmp_path), tmp_path), facts, embedder,
        dimensions=4, embedder_model="toy-embed",
    )
    return loader, embedder, manifest


def _build(tmp_path: Path) -> tuple[MagicMock, MagicMock, BuildManifest]:
    return _build_with(tmp_path, facts_fixture(tmp_path))


def _cypher_params(loader: MagicMock) -> list[dict[str, Any]]:
    return [c.kwargs for c in loader._run.call_args_list]


def test_library_facts_come_from_the_concept_source_import(tmp_path: Path) -> None:
    projection, receipt = write_projection(tmp_path)
    before = hashlib.sha256(projection.read_bytes()).hexdigest()
    facts = import_library_facts(CONCEPT_SPEC, projection, receipt)
    gender = facts.elements["G1"]
    assert (gender.label, gender.definition) == ("Administrative gender", "Gender for billing.")
    assert set(gender.members) == {("L8532", "Female"), ("L8507", "Male")}
    assert set(facts.elements) == set(ELEMENTS)
    assert "toy-library-workload" in facts.release
    detail = FactsConceptSource(facts).name_synonyms("L8532")
    assert detail is not None and detail.name == "Female"
    assert hashlib.sha256(projection.read_bytes()).hexdigest() == before


def test_projection_must_match_its_receipt(tmp_path: Path) -> None:
    projection, receipt = write_projection(tmp_path)
    receipt.write_text(json.dumps({"complete": True, "duckdb": {"sha256": "0" * 64}}))
    with pytest.raises(SpecPinError, match="projection"):
        import_library_facts(CONCEPT_SPEC, projection, receipt)
    receipt.write_text(json.dumps({
        "complete": False,
        "duckdb": {"sha256": hashlib.sha256(projection.read_bytes()).hexdigest()},
    }))
    with pytest.raises(SpecPinError, match="incomplete"):
        import_library_facts(CONCEPT_SPEC, projection, receipt)


def test_build_projects_only_approved_bindings(tmp_path: Path) -> None:
    loader, _, _ = _build(tmp_path)
    props = {
        (p["entity_name"], p["name"]) for p in _cypher_params(loader)
        if "semantic_type" in p
    }
    assert (PEOPLE, "gender_G1") in props
    assert (PEOPLE, "gender_G1_label") in props
    assert (EVENTS, "site_S1") in props
    assert not {name for _, name in props} & {"derived_x", "ghost_col", "age_years"}
    roles = {p["model_role"] for p in _cypher_params(loader) if "model_role" in p}
    assert roles == {"BENCH_S"}


def test_value_sets_hold_every_imported_member(tmp_path: Path) -> None:
    loader, _, manifest = _build(tmp_path)
    anchored = {c.args[1] for c in loader.upsert_value_set.call_args_list}
    assert anchored == {"gender_G1", "vital_V1", "site_S1"}
    members = {c.args[0] for c in loader.add_term_to_value_set.call_args_list}
    assert members == {"L8532", "L8507", "L1", "L2", "P1", "P2", "P3"}


def test_absent_library_elements_are_build_findings(tmp_path: Path) -> None:
    _, _, manifest = _build(tmp_path)
    assert any(f.startswith("library element absent: C1") for f in manifest.findings)
    assert any(f.startswith("library element absent: N1") for f in manifest.findings)


def test_entities_and_properties_are_embedded_with_library_text(
    tmp_path: Path,
) -> None:
    loader, embedder, _ = _build(tmp_path)
    texts = [t for c in embedder.embed_documents.call_args_list for t in c.args[0]]
    assert any(t.startswith("Administrative gender") for t in texts)
    matches = {
        (c.kwargs["label"], tuple(sorted(c.kwargs["match"].items())))
        for c in loader.set_node_embedding.call_args_list
    }
    assert ("Property", (("entity_name", PEOPLE), ("name", "gender_G1"))) in matches
    assert ("Entity", (("name", EVENTS),)) in matches
    loader.create_vector_index.assert_any_call("property_embedding_index", "Property", 4)


def test_join_from_spec_relationship_is_materialized(tmp_path: Path) -> None:
    loader, _, _ = _build(tmp_path)
    loader.upsert_join_path.assert_called_once()
    predicates = str(loader.upsert_join_path.call_args)
    assert "rid_L1" in predicates and PEOPLE in predicates and EVENTS in predicates


def test_graph_identity_tracks_inputs(tmp_path: Path) -> None:
    _, _, first = _build(tmp_path / "a")
    _, _, again = _build(tmp_path / "a")
    assert first.graph_identity == again.graph_identity
    fewer = {**ELEMENTS, "G1": (*ELEMENTS["G1"][:2], ELEMENTS["G1"][2][:1])}
    changed_facts = import_library_facts(
        CONCEPT_SPEC, *write_projection(tmp_path / "b", fewer)
    )
    _, _, changed = _build_with(tmp_path / "a", changed_facts)
    assert changed.graph_identity != first.graph_identity


def test_manifest_round_trips(tmp_path: Path) -> None:
    _, _, manifest = _build(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)
    assert read_manifest(path) == manifest
