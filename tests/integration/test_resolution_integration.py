import pytest
from datetime import datetime, timezone

pytestmark = pytest.mark.integration

from sema.engine.resolution import ResolutionEngine
from sema.graph.loader import GraphLoader
from sema.models.assertions import (
    Assertion, AssertionPredicate, AssertionStatus,
)


def _a(subject, predicate, payload=None, source="llm_interpretation",
       confidence=0.8, status=AssertionStatus.AUTO, run_id="run-1",
       object_ref=None):
    return Assertion(
        id=f"a-{hash(subject + predicate.value + source + run_id) % 100000}",
        subject_ref=subject, predicate=predicate,
        payload=payload or {}, object_ref=object_ref,
        source=source, confidence=confidence,
        status=status, run_id=run_id,
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _count(driver, label):
    with driver.session() as s:
        return s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]


def _count_rels(driver, rel_type):
    with driver.session() as s:
        return s.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c").single()["c"]


@pytest.fixture
def resolver(clean_neo4j):
    loader = GraphLoader(clean_neo4j)
    # Pre-create physical nodes that resolution will reference
    loader.upsert_catalog("cdm")
    loader.upsert_schema("clinical", "cdm")
    loader.upsert_table("cancer_diagnosis", "clinical", "cdm")
    loader.upsert_column("dx_type_cd", "cancer_diagnosis", "clinical", "cdm",
                        data_type="STRING", nullable=True)
    return ResolutionEngine(loader), clean_neo4j


class TestFullResolution:
    def test_entity_property_term_synonym_created(self, resolver):
        engine, driver = resolver
        assertions = [
            _a("unity://cdm.clinical.cancer_diagnosis",
               AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "Cancer Diagnosis", "description": "Primary dx"}),
            _a("unity://cdm.clinical.cancer_diagnosis.dx_type_cd",
               AssertionPredicate.HAS_PROPERTY_NAME,
               {"value": "Diagnosis Type", "description": "Cancer type"}),
            _a("unity://cdm.clinical.cancer_diagnosis.dx_type_cd",
               AssertionPredicate.HAS_SEMANTIC_TYPE,
               {"value": "categorical"}),
            _a("unity://cdm.clinical.cancer_diagnosis.dx_type_cd",
               AssertionPredicate.HAS_DECODED_VALUE,
               {"raw": "CRC", "label": "Colorectal Cancer"}),
            _a("unity://cdm.clinical.cancer_diagnosis.dx_type_cd",
               AssertionPredicate.HAS_DECODED_VALUE,
               {"raw": "BRCA", "label": "Breast Cancer"}),
            _a("unity://cdm.clinical.cancer_diagnosis",
               AssertionPredicate.HAS_SYNONYM,
               {"value": "cancer dx"}),
            _a("unity://cdm.clinical.cancer_diagnosis.dx_type_cd",
               AssertionPredicate.PARENT_OF,
               {"parent": "CRC", "child": "COAD"}, source="pattern_match"),
        ]
        engine.resolve(assertions)

        assert _count(driver, "Entity") == 1
        assert _count(driver, "Property") == 1
        assert _count(driver, "Term") >= 2
        assert _count(driver, "ValueSet") == 1
        assert _count(driver, "Alias") >= 1
        assert _count_rels(driver, "ENTITY_ON_TABLE") >= 1
        assert _count_rels(driver, "HAS_PROPERTY") == 1
        assert _count_rels(driver, "MEMBER_OF") >= 2
        assert _count_rels(driver, "STORED_IN") == 1
        assert _count_rels(driver, "REFERS_TO") >= 1
        assert _count_rels(driver, "PARENT_OF") >= 1

    def test_assertions_stored_with_subject_edges(self, resolver):
        engine, driver = resolver
        assertions = [
            _a("unity://cdm.clinical.cancer_diagnosis",
               AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "Test Entity"}),
        ]
        engine.resolve(assertions)

        assert _count(driver, "Assertion") >= 1

    def test_supersession_on_re_resolve(self, resolver):
        engine, driver = resolver

        # First run
        engine.resolve([
            _a("unity://cdm.clinical.cancer_diagnosis",
               AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "Old Name"}, run_id="run-1"),
        ])

        # Second run
        engine.resolve([
            _a("unity://cdm.clinical.cancer_diagnosis",
               AssertionPredicate.HAS_ENTITY_NAME,
               {"value": "New Name"}, run_id="run-2"),
        ])

        with driver.session() as s:
            results = list(s.run(
                "MATCH (a:Assertion) WHERE a.predicate = 'has_entity_name' "
                "RETURN a.status AS status ORDER BY a.run_id"
            ))
        statuses = [r["status"] for r in results]
        assert "superseded" in statuses
        assert "auto" in statuses
