"""A question that names a hierarchy anchor in other words than its label reaches it
through core vector search over the S build's embedded concept Terms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sema.graph.loader import GraphLoader
from showcase.ontology_query_benchmark.sema_bindings import load_sidecar
from showcase.ontology_query_benchmark.sema_build import build_scoped_graph
from tests.integration.test_sema_hierarchy_anchor_integration import ask, filled
from tests.unit.ontology_query_benchmark.hierarchy_fixture import hierarchy, warehouse
from tests.unit.ontology_query_benchmark.test_sema_bindings import write_fixture
from tests.unit.ontology_query_benchmark.test_sema_build import _catalog
from tests.unit.ontology_query_benchmark.test_sema_hierarchy import _facts, _reader
from tests.unit.ontology_query_benchmark.test_sema_hierarchy_anchor import EVENTS_SQL, members

pytestmark = pytest.mark.integration

ANCHOR_LABEL = "Neoplasm"
QUESTION = "Which events are a kind of tumour?"


class SynonymEmbedder:
    """Puts the anchor's label and a question about tumours on one axis, all else on
    another, so only a vector hit can join the question to the anchor."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        near = text == ANCHOR_LABEL or "tumour" in text
        return [1.0, 0.0, 0.0, 0.0] if near else [0.0, 0.0, 0.0, 1.0]


def test_an_anchor_named_in_other_words_is_found_by_vector_search(
    clean_neo4j: Any, tmp_path: Path,
) -> None:
    facts = _facts(tmp_path, hierarchy())
    manifest = build_scoped_graph(
        GraphLoader(clean_neo4j), _catalog(), load_sidecar(write_fixture(tmp_path), tmp_path),
        facts, SynonymEmbedder(), dimensions=4, embedder_model="toy-embed",
        relation_reader=_reader(),
    )

    text = ask(clean_neo4j, tmp_path, facts, manifest, QUESTION, embedder=SynonymEmbedder())

    assert ANCHOR_LABEL.lower() not in QUESTION.lower()
    assert "anchor Neoplasm = id C1 in std" in text
    assert members(warehouse(), EVENTS_SQL, filled(text, "C1")) == {"L10", "L11"}
    assert manifest.counts["embedded_terms"] == 7
