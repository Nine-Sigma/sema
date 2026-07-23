"""S2-05 integration: the authored person FK becomes a retrievable
``:JoinPath`` (M5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sema.graph.loader import GraphLoader
from sema.graph.physical_catalog import (
    PhysicalColumn,
    StaticPhysicalCatalogSource,
)
from sema.graph.queries import CypherQueries
from sema.graph.target_join_paths import materialize_target_join_paths
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


def test_person_join_path_is_retrievable(clean_neo4j) -> None:
    _load(clean_neo4j)
    with clean_neo4j.session() as s:
        rows = list(s.run(
            CypherQueries.find_join_paths(),
            table_names=["condition_occurrence"],
        ))
    assert len(rows) == 1
    predicates = json.loads(rows[0]["join_predicates"])
    assert predicates == [{
        "left": f"{_CATALOG}.{_SCHEMA}.condition_occurrence.person_id",
        "right": f"{_CATALOG}.{_SCHEMA}.person.person_id",
        "op": "=",
    }]
