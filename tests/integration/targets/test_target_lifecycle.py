"""S2-08 integration (D4): re-load is idempotent in-place; scope-delete
removes the target layer while preserving the shared concept spine."""

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
from sema.graph.target_join_paths import materialize_target_join_paths
from sema.graph.target_lifecycle import delete_target_layer
from sema.graph.target_retrieval_loader import (
    EntityTableBinding,
    PhysicalSchemaBinding,
    project_target_to_retrieval_graph,
)
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


class _FakeConcepts:
    def name_synonyms(self, code: str) -> ConceptDetail | None:
        return ConceptDetail(code, "Glioblastoma multiforme")

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
    materialize_target_join_paths(loader, normalized, binding)
    materialize_concept_value_set(
        loader,
        column=ValueSetColumn(
            _CATALOG, _SCHEMA, "condition_occurrence", "condition_concept_id",
        ),
        concept_vocabulary="OMOP", concept_codes=[_CONCEPT_CODE],
        concepts=_FakeConcepts(), source_schema=_SCHEMA,
    )
    write_value_bridge(
        loader, source_vocabulary="OncoTree", source_code="GBM",
        concept_vocabulary="OMOP", concept_code=_CONCEPT_CODE,
        source_schema="cbioportal_gbm",
    )


def _counts(driver) -> dict[str, int]:
    labels = ["Entity", "Property", "Table", "Column", "JoinPath", "ValueSet"]
    with driver.session() as s:
        return {
            label: s.run(f"MATCH (n:{label}) RETURN count(n) AS n").single()["n"]
            for label in labels
        }


def test_reload_is_count_stable(clean_neo4j) -> None:
    _load(clean_neo4j)
    first = _counts(clean_neo4j)
    _load(clean_neo4j)
    assert _counts(clean_neo4j) == first


def test_delete_removes_layer_preserves_concept_spine(clean_neo4j) -> None:
    _load(clean_neo4j)
    delete_target_layer(
        GraphLoader(clean_neo4j), schema_name=_SCHEMA, catalog=_CATALOG
    )
    with clean_neo4j.session() as s:
        target_entities = s.run(
            "MATCH (e:Entity {model_role: 'TARGET'}) RETURN count(e) AS n"
        ).single()["n"]
        concept = s.run(
            "MATCH (t:Term {code: $c}) RETURN count(t) AS n", c=_CONCEPT_CODE
        ).single()["n"]
        bridge = s.run(
            "MATCH ()-[r:MAPS_TO_CONCEPT]->() RETURN count(r) AS n"
        ).single()["n"]
    assert target_entities == 0
    assert concept == 1
    assert bridge == 1
