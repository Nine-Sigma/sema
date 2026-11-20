"""Question text resolves a hierarchy anchor in arm S's graph, and the anchor-filled
fragment it renders executes in DuckDB. Synthetic two-vocabulary fixture and ACRIS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sema.graph.loader import GraphLoader
from sema.pipeline.retrieval import RetrievalEngine, RetrievalScope
from showcase.ontology_query_benchmark.sema_bindings import (
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.ontology_query_benchmark.sema_build import BuildManifest, build_scoped_graph
from showcase.ontology_query_benchmark.sema_build_utils import LibraryFacts
from showcase.ontology_query_benchmark.sema_context import SContextProvider, anchor_bounds
from tests.unit.ontology_query_benchmark import sema_hierarchy_acris_env as acris
from tests.unit.ontology_query_benchmark.hierarchy_fixture import hierarchy, warehouse
from tests.unit.ontology_query_benchmark.sema_bindings_env import (
    CONVENTIONS,
    catalog_fixture,
    write_fixture,
)
from tests.unit.ontology_query_benchmark.sema_build_env import _catalog, _embedder
from tests.unit.ontology_query_benchmark.sema_context_env import LIMITS
from tests.unit.ontology_query_benchmark.sema_hierarchy_env import RELATIONS, _facts, _reader
from tests.unit.ontology_query_benchmark.sema_hierarchy_anchor_env import (
    EVENTS_SQL,
    deeper_facts,
    deeper_warehouse,
    fragments,
    members,
)

pytestmark = pytest.mark.integration


class ToyEmbedder:
    """The build's constant 4-dimension vectors, without a mock's ``embed_query``."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.5] * 4 for _ in texts]


def build(driver: Any, tmp_path: Path, facts: LibraryFacts) -> BuildManifest:
    sidecar = load_sidecar(write_fixture(tmp_path), tmp_path)
    return build_scoped_graph(
        GraphLoader(driver), _catalog(), sidecar, facts, _embedder(),
        dimensions=4, embedder_model="toy-embed", relation_reader=_reader(),
    )


def ask(driver: Any, tmp_path: Path, facts: LibraryFacts, manifest: BuildManifest,
        question: str, *, slice_depth: bool = True, embedder: Any = None) -> str:  # fmt: skip
    sidecar = load_sidecar(write_fixture(tmp_path), tmp_path)
    products = load_products(sidecar)
    engine = RetrievalEngine(
        driver, embedder or ToyEmbedder(), embedder_model_name="toy-embed",
        scope=RetrievalScope(model_role=sidecar.model_role),
        hierarchy_anchor=anchor_bounds(manifest) if slice_depth else None,
    )
    provider = SContextProvider(
        engine, sidecar, products, resolve_bindings(sidecar, products, catalog_fixture()),
        facts, graph_identity=manifest.graph_identity, conventions=CONVENTIONS,
        limits=LIMITS, hierarchy_relations=manifest.hierarchy_relations,
    )
    return provider(question).text


def filled(text: str, code: str) -> str:
    return next(f for f in fragments(text, "site_S1") if f"= '{code}')" in f)


def test_question_text_resolves_the_anchor_into_an_executable_fragment(
    clean_neo4j: Any, tmp_path: Path,
) -> None:
    facts = _facts(tmp_path, hierarchy())
    manifest = build(clean_neo4j, tmp_path, facts)

    text = ask(clean_neo4j, tmp_path, facts, manifest, "Which events are a kind of neoplasm?")

    assert manifest.hierarchy_relations == dict.fromkeys(RELATIONS, True)
    assert "anchor Neoplasm = id C1 in std" in text
    assert members(warehouse(), EVENTS_SQL, filled(text, "C1")) == {"L10", "L11"}


def test_an_anchor_deeper_than_the_core_default_needs_the_slice_depth(
    clean_neo4j: Any, tmp_path: Path,
) -> None:
    facts = deeper_facts(tmp_path)
    manifest = build(clean_neo4j, tmp_path, facts)
    question = "Which events are a clinical finding?"

    text = ask(clean_neo4j, tmp_path, facts, manifest, question)
    default = ask(clean_neo4j, tmp_path, facts, manifest, question, slice_depth=False)

    assert manifest.counts["hierarchy_depth"] == 4
    assert members(deeper_warehouse(), EVENTS_SQL, filled(text, "R")) == {"L10", "L11"}
    assert "= id R " not in default


def test_an_acris_class_named_in_the_question_resolves_its_document_types(
    clean_neo4j: Any, tmp_path: Path,
) -> None:
    con = acris._acris_warehouse()
    facts = acris._facts(tmp_path, con)
    manifest = build(clean_neo4j, tmp_path, facts)

    text = ask(clean_neo4j, tmp_path, facts, manifest,
               "How many documents are mortgages and instruments?")  # fmt: skip

    source = "SELECT doc_type AS site_S1 FROM wh.acris.master"
    expected = {r["DOC. TYPE"] for r in acris.ROWS if r["CLASS CODE DESCRIPTION"] == acris.CLASS}
    assert members(con, source, filled(text, acris.CLASS)) == expected and len(expected) == 23
