"""Tests for Section 2-4 of `expand-healthcare-eval-coverage`:
study-scoped stamping, MERGE-key corrections, scoped-delete.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from sema.graph.loader import GraphLoader
from sema.graph.loader_utils import (
    batch_upsert_aliases,
    batch_upsert_entities,
    batch_upsert_join_paths,
    batch_upsert_properties,
    batch_upsert_terms,
    batch_upsert_value_sets,
)
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)

pytestmark = pytest.mark.unit

SCHEMA_BRCA = "cbioportal_brca_tcga_pan_can_atlas_2018"
SCHEMA_MSK = "cbioportal_msk_chord_2024"


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(
        return_value=session,
    )
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


@pytest.fixture
def loader(mock_driver):
    driver, _ = mock_driver
    return GraphLoader(driver)


def _assertion(predicate=AssertionPredicate.HAS_LABEL, **overrides):
    base = dict(
        id="a-1",
        subject_ref="databricks://ws/cat/sch/tbl",
        predicate=predicate,
        payload={"value": "x"},
        source="llm_interpretation",
        confidence=0.9,
        status=AssertionStatus.AUTO,
        run_id="run-1",
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    base.update(overrides)
    return Assertion(**base)


class TestAssertionSourceSchemaStamp:
    def test_assertion_model_has_source_schema_field(self):
        a = _assertion(source_schema=SCHEMA_BRCA)
        assert a.source_schema == SCHEMA_BRCA

    def test_assertion_source_schema_optional(self):
        a = _assertion()
        assert a.source_schema is None

    def test_store_assertion_writes_source_schema(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        a = _assertion()
        loader.store_assertion(a, source_schema=SCHEMA_BRCA)
        cypher = session.run.call_args[0][0]
        params = session.run.call_args[1]
        assert "source_schema" in cypher
        assert params["source_schema"] == SCHEMA_BRCA

    def test_commit_table_assertions_threads_source_schema(
        self, loader, mock_driver,
    ):
        driver, session = mock_driver
        tx = MagicMock()
        session.begin_transaction.return_value = tx
        loader.commit_table_assertions(
            [_assertion()], source_schema=SCHEMA_MSK,
        )
        cypher = tx.run.call_args[0][0]
        kwargs = tx.run.call_args[1]
        assert "source_schema: a.source_schema" in cypher
        assert kwargs["assertions"][0]["source_schema"] == SCHEMA_MSK


class TestEdgeStamping:
    def _row(self, **overrides):
        base = dict(
            name="Patient",
            description=None,
            source="llm_interpretation",
            confidence=0.9,
            entity_name="Patient",
            column_name="age",
            table_name="patient",
            schema_name="sch",
            catalog="cat",
        )
        base.update(overrides)
        return base

    def test_entity_on_table_carries_source_schema(self, loader):
        loader._run = MagicMock()
        batch_upsert_entities(
            loader, [self._row()], source_schema=SCHEMA_BRCA,
        )
        cypher = loader._run.call_args[0][0]
        assert "ENTITY_ON_TABLE" in cypher
        assert "source_schema: r.source_schema" in cypher
        rows = loader._run.call_args[1]["rows"]
        assert rows[0]["source_schema"] == SCHEMA_BRCA

    def test_has_property_and_property_on_column_stamped(self, loader):
        loader._run = MagicMock()
        batch_upsert_properties(
            loader, [self._row()], source_schema=SCHEMA_MSK,
        )
        cypher = loader._run.call_args[0][0]
        assert "HAS_PROPERTY" in cypher
        assert "PROPERTY_ON_COLUMN" in cypher
        assert cypher.count("source_schema: r.source_schema") == 2

    def test_batch_upsert_properties_stamps_implicit_entity_role(
        self, loader,
    ):
        loader._run = MagicMock()
        batch_upsert_properties(
            loader, [self._row()], source_schema=SCHEMA_BRCA,
        )
        cypher = loader._run.call_args[0][0]
        assert "MERGE (e:Entity {name: r.entity_name})" in cypher
        merge_idx = cypher.index("MERGE (e:Entity {name: r.entity_name})")
        has_property_idx = cypher.index("HAS_PROPERTY")
        entity_block = cypher[merge_idx:has_property_idx]
        assert "e.model_role = coalesce(e.model_role, 'SOURCE')" in entity_block
        assert (
            "e.source_id = coalesce(e.source_id, r.source_schema, r.source)"
            in entity_block
        )

    def test_batch_upsert_terms_stamps_source_id_from_schema(
        self, loader,
    ):
        loader._run = MagicMock()
        batch_upsert_terms(
            loader,
            [{
                "code": "0", "label": "neutral",
                "vocabulary_name": "cna_call",
                "source": "llm_interpretation", "confidence": 0.9,
            }],
            source_schema=SCHEMA_BRCA,
        )
        cypher = loader._run.call_args[0][0]
        assert (
            "t.source_id = coalesce(t.source_id, r.source_schema, r.source)"
            in cypher
        )
        rows = loader._run.call_args[1]["rows"]
        assert rows[0]["source_schema"] == SCHEMA_BRCA

    def test_has_value_set_stamped(self, loader):
        loader._run = MagicMock()
        batch_upsert_value_sets(
            loader,
            [{
                "name": "vs", "column_ref": "cat.sch.tbl.col",
                "column_name": "col", "table_name": "tbl",
                "schema_name": "sch", "catalog": "cat",
            }],
            source_schema=SCHEMA_BRCA,
        )
        cypher = loader._run.call_args[0][0]
        assert "HAS_VALUE_SET" in cypher
        assert "source_schema: r.source_schema" in cypher

    def test_member_of_stamped(self, loader, mock_driver):
        _, session = mock_driver
        loader.add_term_to_value_set(
            "TP53", "patient_status_values",
            source_schema=SCHEMA_BRCA,
        )
        cypher = session.run.call_args[0][0]
        params = session.run.call_args[1]
        assert "MEMBER_OF" in cypher
        assert "source_schema: $source_schema" in cypher
        assert params["source_schema"] == SCHEMA_BRCA

    def test_parent_of_stamped(self, loader, mock_driver):
        _, session = mock_driver
        loader.add_term_hierarchy(
            "NEOPLASM", "CARCINOMA", source_schema=SCHEMA_MSK,
        )
        cypher = session.run.call_args[0][0]
        params = session.run.call_args[1]
        assert "PARENT_OF" in cypher
        assert params["source_schema"] == SCHEMA_MSK

    def test_refers_to_stamped(self, loader):
        loader._run = MagicMock()
        batch_upsert_aliases(
            loader,
            [{
                "text": "colon cancer",
                "target_key": "ref",
                "parent_name": "Cancer Diagnosis",
                "parent_entity_name": None,
                "source": "llm", "confidence": 0.8,
                "is_preferred": False, "description": None,
            }],
            ":Entity",
            source_schema=SCHEMA_BRCA,
        )
        cypher = loader._run.call_args[0][0]
        assert "REFERS_TO" in cypher
        assert "source_schema: r.source_schema" in cypher


class TestMergeKeys:
    def test_entity_merge_is_name_only_no_datasource_id(self, loader):
        loader._run = MagicMock()
        batch_upsert_entities(
            loader,
            [{
                "name": "Patient", "description": None,
                "source": "llm", "confidence": 0.9,
                "table_name": "p", "schema_name": "s", "catalog": "c",
            }],
            source_schema=SCHEMA_BRCA,
        )
        cypher = loader._run.call_args[0][0]
        assert "MERGE (e:Entity {name: r.name})" in cypher
        assert "datasource_id" not in cypher
        assert "table_key" not in cypher

    def test_property_merge_is_entity_name_plus_name(self, loader):
        loader._run = MagicMock()
        batch_upsert_properties(
            loader,
            [{
                "name": "age", "entity_name": "Patient",
                "semantic_type": "numeric",
                "source": "llm", "confidence": 0.9,
                "column_name": "age", "table_name": "p",
                "schema_name": "s", "catalog": "c",
            }],
            source_schema=SCHEMA_BRCA,
        )
        cypher = loader._run.call_args[0][0]
        assert (
            "MERGE (p:Property {entity_name: r.entity_name, "
            "name: r.name})"
        ) in cypher

    def test_value_set_merge_is_column_ref(self, loader):
        loader._run = MagicMock()
        batch_upsert_value_sets(
            loader,
            [{
                "name": "vs", "column_ref": "c.s.t.col",
                "column_name": "col", "table_name": "t",
                "schema_name": "s", "catalog": "c",
            }],
            source_schema=SCHEMA_BRCA,
        )
        cypher = loader._run.call_args[0][0]
        assert "MERGE (vs:ValueSet {column_ref: r.column_ref})" in cypher

    def test_join_path_merge_keyed_by_name_and_source_schema(
        self, loader,
    ):
        loader._run = MagicMock()
        batch_upsert_join_paths(
            loader,
            [{
                "name": "patient_to_sample",
                "join_predicates": [{"left_table": "patient"}],
                "hop_count": 1, "source": "fk_detector",
                "confidence": 0.95,
                "sql_snippet": None, "cardinality_hint": None,
            }],
            source_schema=SCHEMA_BRCA,
        )
        cypher = loader._run.call_args[0][0]
        assert (
            "MERGE (jp:JoinPath {name: r.name, "
            "source_schema: r.source_schema})"
        ) in cypher


class TestJoinPathEdgeRequiresSourceSchema:
    def test_uses_requires_source_schema(self, loader):
        with pytest.raises(ValueError, match="source_schema"):
            loader.add_join_path_uses("jp", "tbl_ref")

    def test_entity_links_requires_source_schema(self, loader):
        with pytest.raises(ValueError, match="source_schema"):
            loader.add_join_path_entity_links(
                "jp", "from", "to",
            )

    def test_uses_with_schema_matches_by_name_and_schema(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        loader.add_join_path_uses(
            "jp", "tbl_ref", source_schema=SCHEMA_BRCA,
        )
        cypher = session.run.call_args[0][0]
        assert (
            "MATCH (jp:JoinPath {name: $jp_name, "
            "source_schema: $source_schema})"
        ) in cypher


class TestScopedDelete:
    def test_delete_runs_three_queries(self, loader, mock_driver):
        _, session = mock_driver
        loader.delete_study_scoped(SCHEMA_BRCA)
        calls = [c[0][0] for c in session.run.call_args_list]
        assert any(
            "MATCH ()-[r {source_schema: $schema}]-() DELETE r" in c
            for c in calls
        )
        assert any(
            "MATCH (a:Assertion {source_schema: $schema})" in c
            and "DETACH DELETE a" in c
            for c in calls
        )
        assert any(
            "MATCH (jp:JoinPath {source_schema: $schema})" in c
            and "DETACH DELETE jp" in c
            for c in calls
        )
        for c in session.run.call_args_list:
            if "schema" in c[1]:
                assert c[1]["schema"] == SCHEMA_BRCA

    def test_delete_does_not_touch_shared_concept_nodes(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        loader.delete_study_scoped(SCHEMA_BRCA)
        joined = " ".join(
            c[0][0] for c in session.run.call_args_list
        )
        for shared in (
            ":Entity", ":Term", ":ValueSet", ":Property",
            ":SemanticType", ":Table", ":Column", ":Schema",
        ):
            assert f"DELETE n.{shared}" not in joined
            assert f"DETACH DELETE {shared}" not in joined
