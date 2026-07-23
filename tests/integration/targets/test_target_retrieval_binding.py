"""S2-03 integration: the target-retrieval projection is reachable by
``resolve_physical_mapping`` (M1/M2/G-phys), and is idempotent (D5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.graph.loader import GraphLoader
from sema.graph.physical_catalog import (
    PhysicalColumn,
    StaticPhysicalCatalogSource,
)
from sema.graph.queries import CypherQueries
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


def _int(name: str) -> PhysicalColumn:
    return PhysicalColumn(name, "BIGINT", False)


def _binding() -> PhysicalSchemaBinding:
    return PhysicalSchemaBinding(
        catalog=_CATALOG, schema=_SCHEMA,
        entity_tables=(
            EntityTableBinding("omop.person", "person"),
            EntityTableBinding("omop.condition_occurrence", "condition_occurrence"),
        ),
    )


def _catalog() -> StaticPhysicalCatalogSource:
    return StaticPhysicalCatalogSource({
        (_CATALOG, _SCHEMA, "person"): [_int("person_id")],
        (_CATALOG, _SCHEMA, "condition_occurrence"): [
            _int("condition_occurrence_id"), _int("person_id"),
            _int("condition_concept_id"),
            PhysicalColumn("condition_start_date", "DATE", True),
        ],
    })


def _project(driver) -> None:
    normalized = TargetModelNormalizer.normalize(ManifestTargetAdapter(_MANIFEST))
    project_target_to_retrieval_graph(
        GraphLoader(driver), normalized, _binding(), _catalog()
    )


def test_target_entity_resolves_to_physical_binding(clean_neo4j) -> None:
    _project(clean_neo4j)
    with clean_neo4j.session() as s:
        rows = list(s.run(
            CypherQueries.resolve_physical_mapping(),
            entity_name="condition_occurrence",
        ))
    assert len(rows) == 1
    row = rows[0]
    assert row["table_name"] == "condition_occurrence"
    assert row["schema_name"] == _SCHEMA
    sem = {c["column"]: c["semantic_type"] for c in row["columns"]}
    assert sem["condition_concept_id"] == "categorical"
    assert sem["condition_start_date"] == "temporal"


def test_reload_yields_single_current_binding(clean_neo4j) -> None:
    _project(clean_neo4j)
    _project(clean_neo4j)
    with clean_neo4j.session() as s:
        count = s.run(
            "MATCH (e:Entity {name: 'condition_occurrence', "
            "model_role: 'TARGET'}) RETURN count(e) AS n"
        ).single()["n"]
    assert count == 1
