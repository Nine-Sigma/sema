"""S2-09 integration: the generic composition wires the full retrievable
target graph (projection + joins + value set + bridge)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.graph.concept_source import ConceptAncestor, ConceptDetail
from sema.graph.loader import GraphLoader
from sema.graph.physical_catalog import (
    PhysicalColumn,
    StaticPhysicalCatalogSource,
)
from sema.graph.queries import CypherQueries
from sema.graph.target_graph_build import (
    ConceptFieldSpec,
    ValueBridgeSpec,
    build_target_retrieval_graph,
)
from sema.graph.target_retrieval_loader import (
    EntityTableBinding,
    PhysicalSchemaBinding,
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
_CONCEPT = "4001458"


class _FakeConcepts:
    def name_synonyms(self, code: str) -> ConceptDetail | None:
        return ConceptDetail(code, "Glioblastoma multiforme")

    def ancestors(self, code: str) -> list[ConceptAncestor]:
        return []


def _int(name: str) -> PhysicalColumn:
    return PhysicalColumn(name, "BIGINT", False)


def test_build_composes_retrievable_graph(clean_neo4j) -> None:
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
    build_target_retrieval_graph(
        GraphLoader(clean_neo4j), normalized, binding, catalog,
        concepts=_FakeConcepts(), concept_vocabulary="OMOP",
        concept_fields=[ConceptFieldSpec(
            "omop.condition_occurrence", "condition_concept_id", (_CONCEPT,),
        )],
        value_bridges=[ValueBridgeSpec("OncoTree", "GBM", _CONCEPT)],
        bridge_source_schema="cbioportal_gbm",
    )

    engine = RetrievalEngine(clean_neo4j)
    assert engine.resolve_concept_for_source_term("GBM")[0]["concept_code"] == _CONCEPT
    with clean_neo4j.session() as s:
        joins = list(s.run(
            CypherQueries.find_join_paths(),
            table_names=["condition_occurrence"],
        ))
    assert len(joins) == 1
