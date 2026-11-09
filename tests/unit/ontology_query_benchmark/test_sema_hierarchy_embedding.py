"""The S build embeds the slice's named hierarchy concept Terms through core's Term
embedding path, under a named cap, and counts them in the manifest."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from showcase.ontology_query_benchmark import sema_build, sema_hierarchy
from showcase.ontology_query_benchmark.sema_bindings import load_sidecar
from showcase.ontology_query_benchmark.sema_build import BuildManifest, build_scoped_graph
from showcase.ontology_query_benchmark.sema_build_utils import LibraryFacts
from tests.unit.ontology_query_benchmark.hierarchy_fixture import hierarchy
from tests.unit.ontology_query_benchmark.test_sema_bindings import write_fixture
from tests.unit.ontology_query_benchmark.test_sema_build import _catalog, _embedder
from tests.unit.ontology_query_benchmark.test_sema_hierarchy import _facts, _reader

pytestmark = pytest.mark.unit


def _graph_terms(facts: LibraryFacts) -> list[dict[str, str]]:
    """Term nodes as the graph holds them after the build's writers: the slice's
    concepts plus a value-set member Term in the member vocabulary."""
    assert facts.hierarchy is not None
    concepts = [{"vocabulary_name": "std", "code": code, "label": name, "model_role": "SOURCE"}
                for code, name in facts.hierarchy.names.items()]  # fmt: skip
    member = {"vocabulary_name": "lib", "code": "L10", "label": "Member", "model_role": "X"}
    return [*concepts, member]


def _build(tmp_path: Path, facts: LibraryFacts) -> tuple[MagicMock, MagicMock, BuildManifest]:
    loader, embedder = MagicMock(), _embedder()
    loader.query_nodes_by_label.side_effect = (
        lambda label: _graph_terms(facts) if label == "Term" else []
    )
    manifest = build_scoped_graph(
        loader, _catalog(), load_sidecar(write_fixture(tmp_path), tmp_path), facts, embedder,
        dimensions=4, embedder_model="toy-embed", relation_reader=_reader(),
    )
    return loader, embedder, manifest


def _term_embeddings(loader: MagicMock) -> list[dict[str, str]]:
    return [
        dict(c.kwargs["match"])
        for c in loader.set_node_embedding.call_args_list
        if c.kwargs["label"] == "Term"
    ]


def _texts(embedder: MagicMock) -> list[str]:
    return [t for c in embedder.embed_documents.call_args_list for t in c.args[0]]


def test_each_named_hierarchy_concept_is_embedded_by_its_name(tmp_path: Path) -> None:
    facts = _facts(tmp_path, hierarchy())
    loader, embedder, manifest = _build(tmp_path, facts)

    assert facts.hierarchy is not None
    names = dict(facts.hierarchy.names)
    assert len(names) == 7 and names["C1"] == "Neoplasm"
    assert _term_embeddings(loader) == [
        {"vocabulary_name": "std", "code": code, "model_role": "SOURCE"} for code in sorted(names)
    ]
    assert set(names.values()) <= set(_texts(embedder)) and "Member" not in _texts(embedder)
    assert manifest.counts["embedded_terms"] == 7


def test_without_a_hierarchy_no_term_is_embedded_and_no_count_is_recorded(
    tmp_path: Path,
) -> None:
    loader, _, manifest = _build(tmp_path, _facts(tmp_path))

    assert _term_embeddings(loader) == []
    assert "embedded_terms" not in manifest.counts


def test_terms_beyond_the_cap_are_not_embedded_and_the_build_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sema_hierarchy, "TERM_EMBEDDING_CAP", 3)

    loader, _, manifest = _build(tmp_path, _facts(tmp_path, hierarchy()))

    assert len(_term_embeddings(loader)) == manifest.counts["embedded_terms"] == 3
    assert any("7 hierarchy concepts" in f and "cap of 3" in f
               for f in manifest.findings)  # fmt: skip


def test_within_the_cap_the_build_records_no_cap_finding(tmp_path: Path) -> None:
    _, _, manifest = _build(tmp_path, _facts(tmp_path, hierarchy()))

    assert not any("cap of" in f for f in manifest.findings)


def test_a_hierarchy_graph_identity_pins_the_embedding_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts, plain = _facts(tmp_path / "h", hierarchy()), _facts(tmp_path / "p")
    before = [_build(tmp_path, f)[2].graph_identity for f in (facts, plain)]
    monkeypatch.setattr(sema_build, "TERM_EMBEDDING_CAP", 3)
    after = [_build(tmp_path, f)[2].graph_identity for f in (facts, plain)]

    assert after[0] != before[0]
    assert after[1] == before[1]
