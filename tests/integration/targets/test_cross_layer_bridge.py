"""S2-06 integration: the value bridge is retrievable (Finding 1).

A query seeded on a SOURCE term resolves to the target concept + governed
column by traversing MAPS_TO_CONCEPT — the warehouse→ontology join.
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
from sema.pipeline.term_expansion_utils import _expand_term_hit
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
_SOURCE_CODE = "GBM"
_SOURCE_VOCAB = "OncoTree"
_CONCEPT_VOCAB = "OMOP"


class _FakeConcepts:
    def name_synonyms(self, code: str) -> ConceptDetail | None:
        if code == _CONCEPT_CODE:
            return ConceptDetail(code, "Glioblastoma multiforme")
        return None

    def ancestors(self, code: str) -> list[ConceptAncestor]:
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
        concept_vocabulary=_CONCEPT_VOCAB, concept_codes=[_CONCEPT_CODE],
        concepts=_FakeConcepts(), source_schema=_SCHEMA,
    )
    write_value_bridge(
        loader, source_vocabulary=_SOURCE_VOCAB, source_code=_SOURCE_CODE,
        concept_vocabulary=_CONCEPT_VOCAB, concept_code=_CONCEPT_CODE,
        source_schema=_SCHEMA,
    )


def test_resolve_concept_for_source_term(clean_neo4j) -> None:
    _load(clean_neo4j)
    rows = RetrievalEngine(clean_neo4j).resolve_concept_for_source_term(
        _SOURCE_CODE
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["concept_code"] == _CONCEPT_CODE
    assert row["column_name"] == "condition_concept_id"
    assert row["table_name"] == "condition_occurrence"
    assert row["schema_name"] == _SCHEMA


def test_source_term_hit_traverses_bridge(clean_neo4j) -> None:
    _load(clean_neo4j)
    engine = RetrievalEngine(clean_neo4j)
    hit = {"code": _SOURCE_CODE, "vocabulary_name": _SOURCE_VOCAB,
           "label": "Glioblastoma", "status": "auto", "confidence": 0.9}
    candidates = _expand_term_hit(engine, hit)
    bridged = [
        c for c in candidates
        if c.get("source") == "retrieval_concept_bridge"
    ]
    assert len(bridged) == 1
    assert bridged[0]["code"] == _CONCEPT_CODE
    assert bridged[0]["column"] == "condition_concept_id"
    assert bridged[0]["bridged_from"] == _SOURCE_CODE


def test_no_dangling_maps_to_concept(clean_neo4j) -> None:
    _load(clean_neo4j)
    with clean_neo4j.session() as s:
        total = s.run(
            "MATCH ()-[r:MAPS_TO_CONCEPT]->() RETURN count(r) AS n"
        ).single()["n"]
        closed = s.run(
            "MATCH (a:Term)-[r:MAPS_TO_CONCEPT]->(b:Term) RETURN count(r) AS n"
        ).single()["n"]
    assert total == closed == 1
