"""Shared setup for test_sema_hierarchy.py and the tests that reuse it."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from sema.pipeline.dedup_utils import tokenize_query
from showcase.ontology_query_benchmark.profile import Attribution, ConceptHierarchy, ConceptRelation
from showcase.ontology_query_benchmark.sema_bindings import (
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.ontology_query_benchmark.sema_build import BuildManifest, build_scoped_graph
from showcase.ontology_query_benchmark.sema_build_utils import LibraryFacts, import_library_facts
from showcase.ontology_query_benchmark.sema_context import SContext, SContextProvider
from showcase.ontology_query_benchmark.sema_hierarchy_sql import membership_sql
from tests.unit.ontology_query_benchmark.hierarchy_fixture import MEMBERS, add_hierarchy_slice
from tests.unit.ontology_query_benchmark.sema_bindings_env import (
    CONCEPT_SPEC,
    CONVENTIONS,
    ELEMENTS,
    catalog_fixture,
    write_fixture,
    write_projection,
)
from tests.unit.ontology_query_benchmark.sema_build_env import _catalog, _embedder
from tests.unit.ontology_query_benchmark.sema_context_env import EVENTS_ENTITY, LIMITS, _Retriever

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


def _context(tmp_path: Path, facts: LibraryFacts, readable: dict[str, bool] | None) -> SContext:
    sidecar = load_sidecar(write_fixture(tmp_path), tmp_path)
    products = load_products(sidecar)
    provider = SContextProvider(
        _Retriever([EVENTS_ENTITY]), sidecar, products,
        resolve_bindings(sidecar, products, catalog_fixture()), facts,
        graph_identity="graph-1", conventions=CONVENTIONS, limits=LIMITS,
        hierarchy_relations=readable,
    )
    return provider(asking(facts))


def asking(facts: LibraryFacts, labels: Iterable[str] = ()) -> str:
    """A question naming a concept of the facts' hierarchy that shares no word with
    ``labels``, the retrieved anchors, so they rank as for a question naming none."""
    if facts.hierarchy is None:
        return "Which events sit under a broader concept?"
    taken = {w for label in labels for w in tokenize_query(label)}
    concept = next(n for n in sorted(facts.hierarchy.names.values())
                   if not taken & set(tokenize_query(n)))  # fmt: skip
    return f"Which events sit under {concept}?"


def _fragment(text: str, column: str) -> str:
    lines = [x.strip().removeprefix("WHERE ") for x in text.splitlines()]
    return next(x for x in lines if "{column} IN" in x).replace("{column}", column)


NAME_KEYED_PARAMETERS = {"name", "vocabulary", "concept_class"}
QUALIFIER_COLUMNS = {"vocabulary_column", "class_column"}


def require_name_keyed_template() -> None:
    """The template API the name-keyed predicate needs; absent, the red is an assertion."""
    missing = NAME_KEYED_PARAMETERS - set(inspect.signature(membership_sql).parameters)
    assert not missing, f"membership_sql has no name-keyed parameters {sorted(missing)}"
    absent = QUALIFIER_COLUMNS - set(ConceptRelation.model_fields)
    assert not absent, f"ConceptRelation has no qualifier columns {sorted(absent)}"
