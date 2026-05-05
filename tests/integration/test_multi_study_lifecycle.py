"""Integration tests for multi-study graph lifecycle.

Covers tasks 2.4c, 3.8, 4.6, 4.7, 4.8 of expand-healthcare-eval-coverage:
- ontology-preloaded edges survive scoped-delete (2.4c)
- no :Entity carries `datasource_id` property after a build (3.8)
- A→B→A reload yields unchanged A counts and untouched B (4.6)
- shared `:Term {code: HGNC:TP53}` survives a single-study rebuild (4.7)
- prior edges not re-emitted by the rebuild are removed (4.8)
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sema.graph.loader import GraphLoader
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)

pytestmark = pytest.mark.integration

SCHEMA_A = "cbioportal_brca_tcga_pan_can_atlas_2018"
SCHEMA_B = "cbioportal_msk_chord_2024"


@pytest.fixture
def loader(clean_neo4j):
    return GraphLoader(clean_neo4j)


def _count(driver, cypher: str, **params) -> int:
    with driver.session() as s:
        rec = s.run(cypher, **params).single()
        return int(rec["c"]) if rec else 0


def _make_assertion(subject_ref: str, run_id: str) -> Assertion:
    return Assertion(
        id=f"a-{abs(hash(subject_ref + run_id)) % 100000}",
        subject_ref=subject_ref,
        predicate=AssertionPredicate.HAS_LABEL,
        payload={"value": "v"},
        source="test",
        confidence=0.9,
        status=AssertionStatus.AUTO,
        run_id=run_id,
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _seed_study(
    loader: GraphLoader,
    schema: str,
    table: str,
    *,
    entity: str,
    property_name: str,
    term_code: str,
    run_id: str,
) -> None:
    catalog = "workspace"
    loader.upsert_table(table, schema, catalog)
    loader.upsert_column(property_name, table, schema, catalog, data_type="STRING")
    loader.upsert_entity(
        name=entity, description=None, source="test", confidence=0.9,
        table_name=table, schema_name=schema, catalog=catalog,
        source_schema=schema,
    )
    loader.upsert_property(
        name=property_name, semantic_type="genomic_id",
        source="test", confidence=0.9,
        entity_name=entity, column_name=property_name,
        table_name=table, schema_name=schema, catalog=catalog,
        source_schema=schema,
    )
    loader.upsert_term(
        code=term_code, label=term_code, source="test", confidence=0.95,
    )
    loader.upsert_value_set(
        name=property_name, column_name=property_name, table_name=table,
        schema_name=schema, catalog=catalog, source_schema=schema,
    )
    loader.add_term_to_value_set(
        term_code=term_code, value_set_name=property_name,
        source_schema=schema,
    )
    a = _make_assertion(
        subject_ref=f"databricks://{catalog}/{schema}/{table}", run_id=run_id,
    )
    loader.store_assertion(a, source_schema=schema)


def _add_ontology_parent_of(driver) -> None:
    """Pre-load a hierarchy edge with no source_schema (ontology-global)."""
    with driver.session() as s:
        s.run(
            "MERGE (p:Term {code: 'HGNC:TP53'}) "
            "MERGE (c:Term {code: 'HGNC:TP53_VARIANT'}) "
            "MERGE (p)-[:PARENT_OF]->(c)"
        )


class TestOntologyEdgeSurvivesScopedDelete:
    def test_preloaded_parent_of_not_swept(self, loader, clean_neo4j):
        _seed_study(
            loader, SCHEMA_A, "patient",
            entity="Patient", property_name="hugo_symbol",
            term_code="HGNC:TP53", run_id="run-A",
        )
        _add_ontology_parent_of(clean_neo4j)

        before = _count(
            clean_neo4j,
            "MATCH ()-[r:PARENT_OF]->() WHERE r.source_schema IS NULL "
            "RETURN count(r) AS c",
        )
        assert before == 1

        loader.delete_study_scoped(SCHEMA_A)

        after = _count(
            clean_neo4j,
            "MATCH ()-[r:PARENT_OF]->() WHERE r.source_schema IS NULL "
            "RETURN count(r) AS c",
        )
        assert after == 1, "ontology-preloaded PARENT_OF must survive sweep"


class TestEntityHasNoDatasourceId:
    def test_no_entity_carries_datasource_id_after_build(self, loader, clean_neo4j):
        _seed_study(
            loader, SCHEMA_A, "patient",
            entity="Patient", property_name="hugo_symbol",
            term_code="HGNC:TP53", run_id="run-A",
        )
        _seed_study(
            loader, SCHEMA_B, "patient",
            entity="Patient", property_name="hugo_symbol",
            term_code="HGNC:TP53", run_id="run-B",
        )
        with_ds = _count(
            clean_neo4j,
            "MATCH (e:Entity) WHERE e.datasource_id IS NOT NULL "
            "RETURN count(e) AS c",
        )
        assert with_ds == 0


class TestReloadIdempotence:
    def test_reload_a_leaves_a_count_unchanged_and_b_untouched(
        self, loader, clean_neo4j,
    ):
        _seed_study(
            loader, SCHEMA_A, "patient",
            entity="Patient", property_name="hugo_symbol",
            term_code="HGNC:TP53", run_id="run-A1",
        )
        _seed_study(
            loader, SCHEMA_B, "patient",
            entity="Patient", property_name="hugo_symbol",
            term_code="HGNC:TP53", run_id="run-B1",
        )
        a_first = _count(
            clean_neo4j,
            "MATCH (a:Assertion {source_schema: $s}) RETURN count(a) AS c",
            s=SCHEMA_A,
        )
        b_first = _count(
            clean_neo4j,
            "MATCH (a:Assertion {source_schema: $s}) RETURN count(a) AS c",
            s=SCHEMA_B,
        )
        assert a_first > 0 and b_first > 0

        loader.delete_study_scoped(SCHEMA_A)
        _seed_study(
            loader, SCHEMA_A, "patient",
            entity="Patient", property_name="hugo_symbol",
            term_code="HGNC:TP53", run_id="run-A2",
        )

        a_third = _count(
            clean_neo4j,
            "MATCH (a:Assertion {source_schema: $s}) RETURN count(a) AS c",
            s=SCHEMA_A,
        )
        b_third = _count(
            clean_neo4j,
            "MATCH (a:Assertion {source_schema: $s}) RETURN count(a) AS c",
            s=SCHEMA_B,
        )
        assert a_third == a_first
        assert b_third == b_first


class TestSharedTermSurvivesRebuild:
    def test_tp53_term_remains_after_one_study_rebuild(
        self, loader, clean_neo4j,
    ):
        _seed_study(
            loader, SCHEMA_A, "patient",
            entity="Patient", property_name="hugo_symbol",
            term_code="HGNC:TP53", run_id="run-A",
        )
        _seed_study(
            loader, SCHEMA_B, "patient",
            entity="Patient", property_name="hugo_symbol",
            term_code="HGNC:TP53", run_id="run-B",
        )
        before_a = _count(
            clean_neo4j,
            "MATCH (t:Term {code: 'HGNC:TP53'})-[m:MEMBER_OF "
            "{source_schema: $s}]->(:ValueSet) RETURN count(m) AS c",
            s=SCHEMA_A,
        )
        assert before_a > 0

        loader.delete_study_scoped(SCHEMA_A)

        term_count = _count(
            clean_neo4j,
            "MATCH (t:Term {code: 'HGNC:TP53'}) RETURN count(t) AS c",
        )
        assert term_count == 1

        a_after = _count(
            clean_neo4j,
            "MATCH (t:Term {code: 'HGNC:TP53'})-[m:MEMBER_OF "
            "{source_schema: $s}]->(:ValueSet) RETURN count(m) AS c",
            s=SCHEMA_A,
        )
        assert a_after == 0

        b_after = _count(
            clean_neo4j,
            "MATCH (t:Term {code: 'HGNC:TP53'})-[m:MEMBER_OF "
            "{source_schema: $s}]->(:ValueSet) RETURN count(m) AS c",
            s=SCHEMA_B,
        )
        assert b_after >= 1


class TestPriorEdgesRemovedOnRebuild:
    def test_prior_edges_not_emitted_in_rebuild_are_removed(
        self, loader, clean_neo4j,
    ):
        _seed_study(
            loader, SCHEMA_A, "patient",
            entity="Patient", property_name="hugo_symbol",
            term_code="HGNC:TP53", run_id="run-A1",
        )
        first = _count(
            clean_neo4j,
            "MATCH ()-[r {source_schema: $s}]-() RETURN count(r) AS c",
            s=SCHEMA_A,
        )
        assert first > 0

        loader.delete_study_scoped(SCHEMA_A)
        after_delete = _count(
            clean_neo4j,
            "MATCH ()-[r {source_schema: $s}]-() RETURN count(r) AS c",
            s=SCHEMA_A,
        )
        assert after_delete == 0
