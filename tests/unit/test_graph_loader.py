import json
import pytest
from unittest.mock import MagicMock, call
from datetime import datetime, timezone

pytestmark = pytest.mark.unit

from sema.graph.loader import GraphLoader
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(
        return_value=session
    )
    driver.session.return_value.__exit__ = MagicMock(
        return_value=False
    )
    return driver, session


@pytest.fixture
def loader(mock_driver):
    driver, _ = mock_driver
    return GraphLoader(driver)


def _make_assertion(
    subject, predicate, payload=None, object_ref=None,
    source="test", confidence=0.9,
    status=AssertionStatus.AUTO, run_id="run-1",
):
    return Assertion(
        id=f"a-{subject}-{predicate.value}",
        subject_ref=subject,
        predicate=predicate,
        payload=payload or {},
        object_ref=object_ref,
        source=source,
        confidence=confidence,
        status=status,
        run_id=run_id,
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


class TestUpsertDataSource:
    def test_upsert_datasource(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_datasource(
            "ds-1", "databricks://ws", "databricks", "ws",
        )
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":DataSource" in cypher
        assert "ON CREATE SET" in cypher


class TestUpsertPhysicalNodes:
    def test_upsert_catalog(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_catalog("cdm")
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":Catalog" in cypher
        assert "ON CREATE SET" in cypher

    def test_upsert_schema(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_schema("clinical", "cdm")
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":Schema" in cypher
        assert "IN_CATALOG" in cypher

    def test_upsert_table(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_table(
            "cancer_diagnosis", "clinical", "cdm",
            table_type="TABLE", comment="Records",
        )
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":Table" in cypher
        assert "IN_SCHEMA" in cypher
        assert "ON CREATE SET" in cypher

    def test_upsert_column(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_column(
            "dx_type_cd", "cancer_diagnosis", "clinical",
            "cdm", data_type="STRING",
        )
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":Column" in cypher
        assert "IN_TABLE" in cypher
        assert "ON CREATE SET" in cypher


class TestUpsertSemanticNodes:
    def test_upsert_entity_uses_entity_on_table(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        loader.upsert_entity(
            "Cancer Diagnosis", description="Primary dx",
            source="llm", confidence=0.8,
            table_name="cancer_diagnosis",
            schema_name="clinical", catalog="cdm",
        )
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "ENTITY_ON_TABLE" in cypher
        assert "IMPLEMENTED_BY" not in cypher
        assert "ON CREATE SET" in cypher

    def test_upsert_property_uses_property_on_column(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        loader.upsert_property(
            "Diagnosis Type", semantic_type="categorical",
            source="llm", confidence=0.8,
            entity_name="Cancer Diagnosis",
            column_name="dx_type_cd",
            table_name="cancer_diagnosis",
            schema_name="clinical", catalog="cdm",
        )
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "PROPERTY_ON_COLUMN" in cypher
        assert "IMPLEMENTED_BY" not in cypher
        assert "ON CREATE SET" in cypher

    def test_upsert_term(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_term(
            "CRC", "Colorectal Cancer", source="llm",
            confidence=0.85,
        )
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":Term" in cypher
        assert "ON CREATE SET" in cypher

    def test_upsert_value_set(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_value_set(
            "oncotree_types", column_name="dx_type_cd",
            table_name="cancer_diagnosis",
            schema_name="clinical", catalog="cdm",
        )
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":ValueSet" in cypher
        assert "HAS_VALUE_SET" in cypher
        assert "ON CREATE SET" in cypher


class TestAliasOperations:
    def test_upsert_alias(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_alias(
            "colon cancer", parent_label=":Entity",
            parent_name="Cancer Diagnosis",
            source="llm", confidence=0.8,
            is_preferred=False,
            description="Common name",
        )
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":Alias" in cypher
        assert "REFERS_TO" in cypher
        assert "ON CREATE SET" in cypher
        assert "is_preferred" in cypher

    def test_no_synonym_method(self, loader):
        assert not hasattr(loader, "upsert_synonym")

    def test_no_candidate_join_method(self, loader):
        assert not hasattr(loader, "upsert_candidate_join")


class TestJoinPathOperations:
    def test_upsert_join_path(self, loader, mock_driver):
        _, session = mock_driver
        loader.upsert_join_path(
            name="t1/c1=t2/c2",
            join_predicates=[{
                "left_table": "t1", "left_column": "c1",
                "right_table": "t2", "right_column": "c2",
                "operator": "=",
            }],
            hop_count=1, source="heuristic", confidence=0.85,
        )
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert ":JoinPath" in cypher
        assert "ON CREATE SET" in cypher

    def test_add_join_path_entity_links(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        loader.add_join_path_entity_links(
            "t1/c1=t2/c2",
            from_table_ref="databricks://ws/cat/sch/t1",
            to_table_ref="databricks://ws/cat/sch/t2",
        )
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "FROM_ENTITY" in cypher
        assert "TO_ENTITY" in cypher
        assert "ENTITY_ON_TABLE" in cypher


class TestAssertionStorage:
    def test_store_assertion_includes_subject_id(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        assertion = Assertion(
            id="a-1",
            subject_ref="databricks://ws/cdm/clinical/tbl/col",
            subject_id="node-1",
            predicate=AssertionPredicate.HAS_LABEL,
            payload={"value": "My Label"},
            source="test",
            confidence=0.9,
            run_id="run-1",
            observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        loader.store_assertion(assertion)
        calls = [str(c) for c in session.run.call_args_list]
        assert any("subject_id" in c for c in calls)
        assert any("object_id" in c for c in calls)

    def test_no_supersession_mutation_on_store(
        self, loader, mock_driver,
    ):
        """store_assertion no longer mutates prior assertions."""
        _, session = mock_driver
        assertion = _make_assertion(
            "databricks://ws/cdm/clinical/tbl/col",
            AssertionPredicate.HAS_LABEL,
            {"value": "New Label"}, run_id="run-2",
        )
        loader.store_assertion(assertion)
        calls = [str(c) for c in session.run.call_args_list]
        assert not any("superseded" in c for c in calls)


class TestProvenanceEdges:
    def test_materialize_provenance_edges(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        assertion = Assertion(
            id="a-prov",
            subject_ref="databricks://ws/cdm/clinical/tbl",
            subject_id="ent-1",
            predicate=AssertionPredicate.HAS_ENTITY_NAME,
            payload={"value": "Test"},
            object_id="tbl-1",
            source="llm_interpretation",
            confidence=0.9,
            run_id="run-1",
            observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        loader.materialize_provenance_edges([assertion])
        calls = [str(c) for c in session.run.call_args_list]
        assert any("SUBJECT" in c for c in calls)
        assert any("OBJECT" in c for c in calls)

    def test_skips_structural_predicates(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        assertion = Assertion(
            id="a-struct",
            subject_ref="databricks://ws/cdm/clinical/tbl",
            subject_id="tbl-1",
            predicate=AssertionPredicate.TABLE_EXISTS,
            payload={"table_type": "TABLE"},
            source="unity_catalog",
            confidence=1.0,
            run_id="run-1",
            observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        loader.materialize_provenance_edges([assertion])
        session.run.assert_not_called()


class TestVectorIndexes:
    def test_create_vector_indexes_from_config(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        loader.create_vector_indexes_from_config(
            ["Entity", "Property", "Alias"],
        )
        assert session.run.call_count == 3
        calls = [
            session.run.call_args_list[i][0][0]
            for i in range(3)
        ]
        assert any("entity_embedding_index" in c for c in calls)
        assert any("property_embedding_index" in c for c in calls)
        assert any("alias_embedding_index" in c for c in calls)


class TestBatchOperations:
    def test_batch_upsert_columns(self, loader, mock_driver):
        _, session = mock_driver
        columns = [
            {
                "name": "col1", "data_type": "STRING",
                "nullable": True, "comment": None,
                "id": "c1", "ref": "r1",
            },
        ]
        loader.batch_upsert_columns(
            columns, "tbl", "schema", "catalog",
        )
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "UNWIND" in cypher
        assert "ON CREATE SET" in cypher
