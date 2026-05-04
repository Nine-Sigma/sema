"""Integration test 12.15: patient↔sample JoinPath wired by {name, source_schema}.

Asserts that materialize_join_paths produces a `:JoinPath` whose `:USES`,
`:FROM_ENTITY`, and `:TO_ENTITY` edges all match the JoinPath by the full
{name, source_schema} match key — guaranteeing two studies emitting the
same logical join produce two distinct paths.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sema.graph.join_materializer import materialize_join_paths
from sema.graph.loader import GraphLoader
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)

pytestmark = pytest.mark.integration

CATALOG = "workspace"
SCHEMA_A = "cbioportal_brca_tcga_pan_can_atlas_2018"
SCHEMA_B = "cbioportal_msk_chord_2024"


@pytest.fixture
def loader(clean_neo4j):
    return GraphLoader(clean_neo4j)


def _ref(catalog: str, schema: str, table: str, column: str | None = None) -> str:
    base = f"databricks://workspace/{catalog}/{schema}/{table}"
    return f"{base}/{column}" if column else base


def _seed_physical(loader: GraphLoader, schema: str) -> None:
    loader.upsert_table("patient", schema, CATALOG)
    loader.upsert_table("sample", schema, CATALOG)
    loader.upsert_column("patient_id", "patient", schema, CATALOG, "STRING")
    loader.upsert_column("patient_id", "sample", schema, CATALOG, "STRING")
    loader._run(
        "MATCH (t:Table {name: 'patient', schema_name: $s, catalog: $c}) "
        "SET t.ref = $ref",
        s=schema, c=CATALOG, ref=_ref(CATALOG, schema, "patient"),
    )
    loader._run(
        "MATCH (t:Table {name: 'sample', schema_name: $s, catalog: $c}) "
        "SET t.ref = $ref",
        s=schema, c=CATALOG, ref=_ref(CATALOG, schema, "sample"),
    )
    loader._run(
        "MATCH (c:Column {name: 'patient_id', table_name: $t, "
        "schema_name: $s, catalog: $cat}) SET c.ref = $ref",
        t="patient", s=schema, cat=CATALOG,
        ref=_ref(CATALOG, schema, "patient", "patient_id"),
    )
    loader._run(
        "MATCH (c:Column {name: 'patient_id', table_name: $t, "
        "schema_name: $s, catalog: $cat}) SET c.ref = $ref",
        t="sample", s=schema, cat=CATALOG,
        ref=_ref(CATALOG, schema, "sample", "patient_id"),
    )
    loader.upsert_entity(
        name="Patient", description=None, source="test", confidence=0.9,
        table_name="patient", schema_name=schema, catalog=CATALOG,
        source_schema=schema,
    )
    loader.upsert_entity(
        name="Sample", description=None, source="test", confidence=0.9,
        table_name="sample", schema_name=schema, catalog=CATALOG,
        source_schema=schema,
    )


def _join_assertion(schema: str) -> Assertion:
    return Assertion(
        id=f"j-{schema}",
        subject_ref=f"join_evidence://{schema}/patient_sample",
        predicate=AssertionPredicate.HAS_JOIN_EVIDENCE,
        payload={
            "join_predicates": [{
                "left_table": _ref(CATALOG, schema, "sample"),
                "left_column": _ref(CATALOG, schema, "sample", "patient_id"),
                "right_table": _ref(CATALOG, schema, "patient"),
                "right_column": _ref(CATALOG, schema, "patient", "patient_id"),
            }],
            "hop_count": 1,
            "from_table": _ref(CATALOG, schema, "sample"),
            "to_table": _ref(CATALOG, schema, "patient"),
        },
        source="fk_detector",
        confidence=0.95,
        status=AssertionStatus.AUTO,
        run_id="run-1",
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _materialize(loader: GraphLoader, assertion: Assertion, schema: str) -> None:
    groups = {(assertion.subject_ref, assertion.predicate.value): [assertion]}
    materialize_join_paths(loader, groups, source_schema=schema)


class TestJoinPathSourceSchemaScoping:
    def test_two_studies_produce_two_distinct_join_paths(
        self, loader, clean_neo4j,
    ):
        _seed_physical(loader, SCHEMA_A)
        _seed_physical(loader, SCHEMA_B)
        _materialize(loader, _join_assertion(SCHEMA_A), SCHEMA_A)
        _materialize(loader, _join_assertion(SCHEMA_B), SCHEMA_B)

        with clean_neo4j.session() as s:
            jp_count = s.run(
                "MATCH (jp:JoinPath) RETURN count(jp) AS c"
            ).single()["c"]
            schemas = sorted(r["s"] for r in s.run(
                "MATCH (jp:JoinPath) RETURN jp.source_schema AS s"
            ))
        assert jp_count == 2
        assert schemas == [SCHEMA_A, SCHEMA_B]

    def test_uses_and_entity_edges_scoped_by_source_schema(
        self, loader, clean_neo4j,
    ):
        _seed_physical(loader, SCHEMA_A)
        _materialize(loader, _join_assertion(SCHEMA_A), SCHEMA_A)

        with clean_neo4j.session() as s:
            uses = s.run(
                "MATCH (jp:JoinPath {source_schema: $sc})"
                "-[u:USES]->(t:Table) RETURN count(u) AS c, "
                "collect(DISTINCT u.source_schema) AS ss",
                sc=SCHEMA_A,
            ).single()
            from_entity = s.run(
                "MATCH (jp:JoinPath {source_schema: $sc})"
                "-[r:FROM_ENTITY]->(:Entity {name: 'Sample'}) "
                "RETURN count(r) AS c, "
                "collect(DISTINCT r.source_schema) AS ss",
                sc=SCHEMA_A,
            ).single()
            to_entity = s.run(
                "MATCH (jp:JoinPath {source_schema: $sc})"
                "-[r:TO_ENTITY]->(:Entity {name: 'Patient'}) "
                "RETURN count(r) AS c, "
                "collect(DISTINCT r.source_schema) AS ss",
                sc=SCHEMA_A,
            ).single()
        assert uses["c"] >= 2
        assert uses["ss"] == [SCHEMA_A]
        assert from_entity["c"] == 1
        assert from_entity["ss"] == [SCHEMA_A]
        assert to_entity["c"] == 1
        assert to_entity["ss"] == [SCHEMA_A]

    def test_scoped_delete_removes_only_one_studys_join_path(
        self, loader, clean_neo4j,
    ):
        _seed_physical(loader, SCHEMA_A)
        _seed_physical(loader, SCHEMA_B)
        _materialize(loader, _join_assertion(SCHEMA_A), SCHEMA_A)
        _materialize(loader, _join_assertion(SCHEMA_B), SCHEMA_B)

        loader.delete_study_scoped(SCHEMA_A)

        with clean_neo4j.session() as s:
            remaining = sorted(r["s"] for r in s.run(
                "MATCH (jp:JoinPath) RETURN jp.source_schema AS s"
            ))
            uses_left = s.run(
                "MATCH (:JoinPath)-[u:USES]->() RETURN count(u) AS c, "
                "collect(DISTINCT u.source_schema) AS ss"
            ).single()
        assert remaining == [SCHEMA_B]
        assert uses_left["ss"] == [SCHEMA_B]
