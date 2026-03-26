import pytest
from datetime import datetime, timezone

pytestmark = pytest.mark.integration

from sema.engine.structural import StructuralEngine
from sema.graph.loader import GraphLoader
from sema.models.assertions import Assertion, AssertionPredicate


def _make_assertion(subject, predicate, payload=None, object_ref=None):
    return Assertion(
        id=f"a-{hash(subject + predicate.value) % 100000}",
        subject_ref=subject,
        predicate=predicate,
        payload=payload or {},
        object_ref=object_ref,
        source="unity_catalog",
        confidence=1.0,
        run_id="run-1",
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _count(driver, label):
    with driver.session() as s:
        return s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]


def _count_rels(driver, rel_type):
    with driver.session() as s:
        return s.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c").single()["c"]


@pytest.fixture
def engine(clean_neo4j):
    loader = GraphLoader(clean_neo4j)
    return StructuralEngine(loader), clean_neo4j


class TestL1EndToEnd:
    def test_full_extraction_creates_graph(self, engine):
        eng, driver = engine
        assertions = [
            # Two tables
            _make_assertion("unity://cdm.clinical.cancer_diagnosis",
                          AssertionPredicate.TABLE_EXISTS, {"table_type": "TABLE"}),
            _make_assertion("unity://cdm.clinical.cancer_diagnosis",
                          AssertionPredicate.HAS_COMMENT, {"value": "Diagnosis records"}),
            _make_assertion("unity://cdm.clinical.cancer_surgery",
                          AssertionPredicate.TABLE_EXISTS, {"table_type": "TABLE"}),

            # Columns for cancer_diagnosis
            _make_assertion("unity://cdm.clinical.cancer_diagnosis.dx_type_cd",
                          AssertionPredicate.COLUMN_EXISTS,
                          {"data_type": "STRING", "nullable": True, "comment": "Type code"}),
            _make_assertion("unity://cdm.clinical.cancer_diagnosis.patient_id",
                          AssertionPredicate.COLUMN_EXISTS,
                          {"data_type": "STRING", "nullable": False, "comment": None}),

            # Columns for cancer_surgery
            _make_assertion("unity://cdm.clinical.cancer_surgery.procedure_date",
                          AssertionPredicate.COLUMN_EXISTS,
                          {"data_type": "DATE", "nullable": True, "comment": None}),

            # FK join
            _make_assertion("unity://cdm.clinical.cancer_diagnosis",
                          AssertionPredicate.JOINS_TO,
                          {"on_column": "patient_id", "cardinality": "one-to-many"},
                          object_ref="unity://cdm.clinical.cancer_surgery"),
        ]

        eng.process(assertions)

        # Verify counts
        assert _count(driver, "Catalog") == 1
        assert _count(driver, "Schema") == 1
        assert _count(driver, "Table") == 2
        assert _count(driver, "Column") == 3

        # Verify edges
        assert _count_rels(driver, "IN_CATALOG") == 1
        assert _count_rels(driver, "IN_SCHEMA") == 2  # 2 tables in schema
        assert _count_rels(driver, "IN_TABLE") == 3   # 3 columns in tables
        assert _count(driver, "JoinPath") == 1

        # Verify table properties
        with driver.session() as s:
            tbl = s.run(
                "MATCH (t:Table {name: 'cancer_diagnosis'}) RETURN t"
            ).single()["t"]
        assert tbl["comment"] == "Diagnosis records"
        assert tbl["table_type"] == "TABLE"

        # Verify column properties
        with driver.session() as s:
            col = s.run(
                "MATCH (c:Column {name: 'dx_type_cd'}) RETURN c"
            ).single()["c"]
        assert col["data_type"] == "STRING"
        assert col["nullable"] is True

    def test_idempotent_processing(self, engine):
        eng, driver = engine
        assertions = [
            _make_assertion("unity://cdm.clinical.tbl",
                          AssertionPredicate.TABLE_EXISTS, {"table_type": "TABLE"}),
            _make_assertion("unity://cdm.clinical.tbl.col",
                          AssertionPredicate.COLUMN_EXISTS,
                          {"data_type": "STRING", "nullable": True, "comment": None}),
        ]
        eng.process(assertions)
        eng.process(assertions)  # run again

        assert _count(driver, "Table") == 1
        assert _count(driver, "Column") == 1
