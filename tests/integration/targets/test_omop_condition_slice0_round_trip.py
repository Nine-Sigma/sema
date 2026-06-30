"""US-007 integration: load the authored OMOP condition_occurrence manifest
into Neo4j and assert the TARGET binding bridge is queryable.

Skip-guarded — runs only when a Neo4j is reachable. The :Entity node
carries ``qualified_name`` (not ``name``), so the AC's ``e.name`` is read
here as ``e.qualified_name`` (aliased ``name``); the row shape matches the
AC: (condition_occurrence | Condition | true).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.graph.target_loader_migrations import cypher_down, cypher_up
from sema.targets.adapters.manifest import ManifestTargetAdapter
from sema.targets.loader import load_target
from sema.targets.neo4j_writer import Neo4jGraphWriter

pytestmark = pytest.mark.integration

_MANIFEST = (
    Path(__file__).resolve().parents[2].parent
    / "src"
    / "sema"
    / "targets"
    / "manifests"
    / "omop_condition_slice0.yaml"
)


@pytest.fixture
def migrated_neo4j(clean_neo4j):
    with clean_neo4j.session() as session:
        for stmt in cypher_up():
            session.run(stmt)
    yield clean_neo4j
    with clean_neo4j.session() as session:
        for stmt in cypher_down():
            session.run(stmt)


def test_condition_binding_bridge_query_returns_one_row(migrated_neo4j) -> None:
    load_target(ManifestTargetAdapter(_MANIFEST), writer=Neo4jGraphWriter(migrated_neo4j))
    with migrated_neo4j.session() as s:
        rows = list(
            s.run(
                "MATCH (e:Entity {model_role: 'TARGET'})-[:HAS_PROPERTY]->"
                "(p:Property)-[:HAS_VOCABULARY_BINDING]->(vb:VocabularyBinding) "
                "WHERE vb.property_name = 'condition_concept_id' "
                "RETURN e.qualified_name AS name, vb.domain AS domain, "
                "vb.require_standard AS require_standard"
            )
        )
    assert len(rows) == 1
    row = rows[0]
    assert row["name"].endswith("condition_occurrence")
    assert row["domain"] == "Condition"
    assert row["require_standard"] is True


def test_condition_occurrence_has_target_obligation(migrated_neo4j) -> None:
    load_target(ManifestTargetAdapter(_MANIFEST), writer=Neo4jGraphWriter(migrated_neo4j))
    with migrated_neo4j.session() as s:
        rows = list(
            s.run(
                "MATCH (e:Entity {qualified_name: 'omop.condition_occurrence'})"
                "-[:HAS_OBLIGATION]->(o:TargetObligation) "
                "RETURN o.payload_json AS payload"
            )
        )
    assert len(rows) == 1
    assert "condition_concept_id" in rows[0]["payload"]
