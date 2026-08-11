"""M4 across studies: two source studies share ONE concept spine, and each
study's bridge edges reflect only the codes that study actually staged.

The DoD requires a code to resolve to a single canonical `:Term` referenced by
*both* studies' bridge edges. It also requires per-study honesty: writing a
study's bridge from the (study-independent) value-mapping store without scoping
it to observed values asserts that a study maps codes it never contained
(bug-435).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.graph.concept_source import ConceptAncestor, ConceptDetail
from sema.graph.concept_value_set import (
    ValueSetColumn,
    materialize_concept_value_set,
)
from sema.graph.cross_layer_bridge import write_value_bridge
from sema.graph.loader import GraphLoader
from sema.graph.physical_catalog import (
    PhysicalColumn,
    StaticPhysicalCatalogSource,
)
from sema.graph.target_retrieval_loader import (
    EntityTableBinding,
    PhysicalSchemaBinding,
    project_target_to_retrieval_graph,
)
from sema.pipeline.retrieval import RetrievalEngine
from sema.targets.adapters.manifest import ManifestTargetAdapter
from sema.targets.normalizer import TargetModelNormalizer

pytestmark = pytest.mark.integration

_MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "showcase" / "cbioportal_to_omop" / "manifests"
    / "omop_condition_slice0.yaml"
)
_CATALOG = "workspace"
_SCHEMA = "omop_stage_a"
_CONCEPT_VOCAB = "OMOP"
_SOURCE_VOCAB = "OncoTree"

_STUDY_A = "study_a"
_STUDY_B = "study_b"

# The shared code is the point: both studies stage it, so it must collapse to
# one concept Term carrying one bridge edge per study.
_SHARED_CODE = "LUAD"
_SHARED_CONCEPT = "45768916"
_A_ONLY_CODE = "SKCM"
_A_ONLY_CONCEPT = "141232"

_STAGED = {
    _STUDY_A: {_SHARED_CODE: _SHARED_CONCEPT, _A_ONLY_CODE: _A_ONLY_CONCEPT},
    _STUDY_B: {_SHARED_CODE: _SHARED_CONCEPT},
}
_LABELS = {
    _SHARED_CONCEPT: "Primary adenocarcinoma of lung",
    _A_ONLY_CONCEPT: "Malignant melanoma of skin",
}


class _FakeConcepts:
    def name_synonyms(self, code: str) -> ConceptDetail | None:
        label = _LABELS.get(code)
        return ConceptDetail(code, label) if label else None

    def ancestors(self, code: str) -> list[ConceptAncestor]:
        return []


def _int(name: str) -> PhysicalColumn:
    return PhysicalColumn(name, "BIGINT", False)


def _project(loader: GraphLoader) -> None:
    normalized = TargetModelNormalizer.normalize(ManifestTargetAdapter(_MANIFEST))
    binding = PhysicalSchemaBinding(
        catalog=_CATALOG, schema=_SCHEMA,
        entity_tables=(
            EntityTableBinding("omop.person", "person"),
            EntityTableBinding("omop.condition_occurrence", "condition_occurrence"),
        ),
    )
    catalog = StaticPhysicalCatalogSource({
        (_CATALOG, _SCHEMA, "person"): [_int("person_id")],
        (_CATALOG, _SCHEMA, "condition_occurrence"): [
            _int("condition_occurrence_id"), _int("person_id"),
            _int("condition_concept_id"),
            PhysicalColumn("condition_start_date", "DATE", True),
        ],
    })
    project_target_to_retrieval_graph(loader, normalized, binding, catalog)


def _materialize_study(loader: GraphLoader, study: str) -> None:
    """One `sema materialize-target-graph` run for a single study."""
    staged = _STAGED[study]
    materialize_concept_value_set(
        loader,
        column=ValueSetColumn(
            _CATALOG, _SCHEMA, "condition_occurrence", "condition_concept_id",
        ),
        concept_vocabulary=_CONCEPT_VOCAB,
        concept_codes=sorted(set(staged.values())),
        concepts=_FakeConcepts(),
        source_schema=_SCHEMA,
    )
    for source_code, concept_code in staged.items():
        write_value_bridge(
            loader, source_vocabulary=_SOURCE_VOCAB, source_code=source_code,
            concept_vocabulary=_CONCEPT_VOCAB, concept_code=concept_code,
            source_schema=study,
        )


def _load_both(driver) -> None:
    loader = GraphLoader(driver)
    _project(loader)
    _materialize_study(loader, _STUDY_A)
    _materialize_study(loader, _STUDY_B)


def _bridge_counts(driver) -> dict[str, int]:
    with driver.session() as s:
        return {
            r["schema"]: r["n"]
            for r in s.run(
                "MATCH ()-[r:MAPS_TO_CONCEPT]->() "
                "RETURN r.source_schema AS schema, count(r) AS n"
            )
        }


def test_shared_code_collapses_to_one_concept_term(clean_neo4j) -> None:
    _load_both(clean_neo4j)
    with clean_neo4j.session() as s:
        terms = s.run(
            "MATCH (t:Term {code: $c}) RETURN count(t) AS n", c=_SHARED_CONCEPT
        ).single()["n"]
    assert terms == 1


def test_both_studies_bridge_to_the_same_concept_node(clean_neo4j) -> None:
    _load_both(clean_neo4j)
    with clean_neo4j.session() as s:
        rows = list(s.run(
            "MATCH (src:Term {code: $c})-[r:MAPS_TO_CONCEPT]->(concept:Term) "
            "RETURN r.source_schema AS schema, elementId(concept) AS node",
            c=_SHARED_CODE,
        ))
    assert {r["schema"] for r in rows} == {_STUDY_A, _STUDY_B}
    assert len({r["node"] for r in rows}) == 1


def test_bridge_edges_are_scoped_to_codes_each_study_staged(clean_neo4j) -> None:
    _load_both(clean_neo4j)
    assert _bridge_counts(clean_neo4j) == {_STUDY_A: 2, _STUDY_B: 1}


def test_study_b_does_not_bridge_a_code_it_never_staged(clean_neo4j) -> None:
    _load_both(clean_neo4j)
    with clean_neo4j.session() as s:
        leaked = s.run(
            "MATCH (src:Term {code: $c})-[r:MAPS_TO_CONCEPT]->() "
            "WHERE r.source_schema = $study RETURN count(r) AS n",
            c=_A_ONLY_CODE, study=_STUDY_B,
        ).single()["n"]
    assert leaked == 0


def test_shared_concept_resolves_once_for_the_shared_code(clean_neo4j) -> None:
    _load_both(clean_neo4j)
    rows = RetrievalEngine(clean_neo4j).resolve_concept_for_source_term(
        _SHARED_CODE
    )
    assert {r["concept_code"] for r in rows} == {_SHARED_CONCEPT}
    assert {r["schema_name"] for r in rows} == {_SCHEMA}


def test_reload_of_one_study_is_edge_stable(clean_neo4j) -> None:
    _load_both(clean_neo4j)
    before = _bridge_counts(clean_neo4j)
    _materialize_study(GraphLoader(clean_neo4j), _STUDY_A)
    assert _bridge_counts(clean_neo4j) == before
