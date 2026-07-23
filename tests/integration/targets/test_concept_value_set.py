"""S2-04 integration: a Column-anchored concept ValueSet is surfaced by
``_expand_values`` and found by ``_lexical_search`` (M3/M4/G-concept)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.graph.concept_source import ConceptAncestor, ConceptDetail
from sema.graph.concept_value_set import (
    ValueSetColumn,
    materialize_concept_value_set,
)
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
_CONCEPT_CODE = "4001458"
_ANCESTOR_CODE = "4178301"
_CONCEPT_VOCAB = "OMOP"


class _FakeConcepts:
    def name_synonyms(self, code: str) -> ConceptDetail | None:
        if code == _CONCEPT_CODE:
            return ConceptDetail(
                code, "Glioblastoma multiforme", ("GBM", "glioblastoma"),
            )
        return None

    def ancestors(self, code: str) -> list[ConceptAncestor]:
        if code == _CONCEPT_CODE:
            return [ConceptAncestor(_ANCESTOR_CODE, "Malignant neoplasm")]
        return []


def _int(name: str) -> PhysicalColumn:
    return PhysicalColumn(name, "BIGINT", False)


def _load(driver) -> None:
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
    loader = GraphLoader(driver)
    project_target_to_retrieval_graph(loader, normalized, binding, catalog)
    materialize_concept_value_set(
        loader,
        column=ValueSetColumn(
            _CATALOG, _SCHEMA, "condition_occurrence", "condition_concept_id",
        ),
        concept_vocabulary=_CONCEPT_VOCAB,
        concept_codes=[_CONCEPT_CODE],
        concepts=_FakeConcepts(),
        source_schema=_SCHEMA,
    )


def test_expand_values_surfaces_concept(clean_neo4j) -> None:
    _load(clean_neo4j)
    engine = RetrievalEngine(clean_neo4j)
    values = engine.expand_from_entities(["condition_occurrence"])["values"]
    surfaced = {v["code"]: v["label"] for v in values}
    assert surfaced.get(_CONCEPT_CODE) == "Glioblastoma multiforme"


def test_lexical_search_finds_concept_by_name(clean_neo4j) -> None:
    _load(clean_neo4j)
    engine = RetrievalEngine(clean_neo4j)
    hits = engine._lexical_search("glioblastoma")
    labels = [h.get("label", "") for h in hits if h.get("node_type") == "term"]
    assert any("Glioblastoma" in label for label in labels)


def test_ancestor_hierarchy_written(clean_neo4j) -> None:
    _load(clean_neo4j)
    with clean_neo4j.session() as s:
        n = s.run(
            "MATCH (a:Term {code: $anc})-[:PARENT_OF]->(c:Term {code: $code}) "
            "RETURN count(*) AS n",
            anc=_ANCESTOR_CODE, code=_CONCEPT_CODE,
        ).single()["n"]
    assert n == 1
