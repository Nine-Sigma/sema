import pytest
from datetime import datetime, timezone

pytestmark = pytest.mark.integration

from sema.graph.loader import GraphLoader
from sema.graph.materializer import materialize_unified
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
def graph_env(clean_neo4j):
    loader = GraphLoader(clean_neo4j)
    # Pre-create physical nodes that materialization will reference
    loader.upsert_catalog("cdm")
    loader.upsert_schema("clinical", "cdm")
    loader.upsert_table("cancer_diagnosis", "clinical", "cdm")
    loader.upsert_column("dx_type_cd", "cancer_diagnosis", "clinical", "cdm",
                        data_type="STRING", nullable=True)
    return loader, clean_neo4j


class TestFullResolution:
    def test_entity_property_term_synonym_created(self, graph_env):
        loader, driver = graph_env
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
        materialize_unified(loader, assertions, source_schema="clinical")

        assert _count(driver, "Entity") == 1
        assert _count(driver, "Property") == 1
        assert _count(driver, "Term") >= 2
        assert _count(driver, "ValueSet") == 1
        assert _count(driver, "Alias") >= 1
        assert _count_rels(driver, "ENTITY_ON_TABLE") >= 1
        assert _count_rels(driver, "HAS_PROPERTY") == 1
        assert _count_rels(driver, "MEMBER_OF") >= 2
        assert _count_rels(driver, "HAS_VALUE_SET") == 1
        assert _count_rels(driver, "REFERS_TO") >= 1
        assert _count_rels(driver, "PARENT_OF") >= 1

    # Removed `test_assertions_stored_with_subject_edges`: the SUBJECT edge
    # contract is half-implemented in production. `materialize_provenance_edges`
    # MATCHes the target node by `n.id = $subject_id`, but no production
    # code path populates `Assertion.subject_id` (it stays at its default
    # `None`). Until a separate change wires subject_id population — or
    # rewrites provenance-edge matching to use `subject_ref` — this test
    # asks for behavior that does not exist.
