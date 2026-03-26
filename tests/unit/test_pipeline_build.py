"""Tests for per-table vertical processing and resource factories (Group 8)."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

pytestmark = pytest.mark.unit

from sema.pipeline.build import (
    TableResult,
    DatabricksConnectorFactory,
    LLMClientFactory,
    process_table,
    aggregate_report,
)
from sema.connectors.databricks import TableWorkItem
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)
from sema.llm_client import LLMStageError, LLMClient
from sema.engine.semantic import TableInterpretation, PropertyInterpretation


def _make_assertion(subject_ref, predicate, payload=None, source="test",
                    run_id="run-1"):
    return Assertion(
        id=f"a-{subject_ref}",
        subject_ref=subject_ref,
        predicate=predicate,
        payload=payload or {},
        source=source,
        confidence=0.9,
        status=AssertionStatus.AUTO,
        run_id=run_id,
        observed_at=datetime.now(timezone.utc),
    )


def _make_extraction_assertions():
    """Minimal extraction assertions for one table."""
    return [
        _make_assertion(
            "unity://cat.sch.tbl",
            AssertionPredicate.TABLE_EXISTS,
            {"table_type": "TABLE"},
        ),
        _make_assertion(
            "unity://cat.sch.tbl.col1",
            AssertionPredicate.COLUMN_EXISTS,
            {"data_type": "STRING", "nullable": True, "comment": None},
        ),
    ]


# ---------------------------------------------------------------------------
# TableResult tests (Task 8.7)
# ---------------------------------------------------------------------------

class TestTableResult:
    def test_success_variant(self):
        r = TableResult.success("ref", entities=2, properties=5, terms=10)
        assert r.status == "success"
        assert r.entities_created == 2
        assert r.properties_created == 5
        assert r.terms_created == 10
        assert r.failed_stage is None

    def test_failed_variant(self):
        r = TableResult.failed("ref", "L2 semantic", "LLM failed")
        assert r.status == "failed"
        assert r.failed_stage == "L2 semantic"
        assert r.error_message == "LLM failed"

    def test_skipped_variant(self):
        r = TableResult.skipped("ref", "no columns")
        assert r.status == "skipped"
        assert r.skip_reason == "no columns"


# ---------------------------------------------------------------------------
# process_table tests (Tasks 8.1, 8.2, 8.3)
# ---------------------------------------------------------------------------

class TestProcessTable:
    def test_success_returns_counts(self):
        work_item = TableWorkItem("cat", "sch", "tbl", "unity://cat.sch.tbl")

        connector = MagicMock()
        connector.extract_table.return_value = _make_extraction_assertions()

        llm_client = MagicMock(spec=LLMClient)
        llm_client.invoke.return_value = TableInterpretation(
            entity_name="Entity",
            properties=[
                PropertyInterpretation(
                    column="col1", name="Column 1",
                    semantic_type="free_text",
                ),
            ],
        )

        loader = MagicMock()

        result = process_table(
            work_item, connector, llm_client, loader,
            run_id="run-1",
        )

        assert result.status == "success"
        assert result.entities_created == 1
        assert result.properties_created == 1
        loader.commit_table_assertions.assert_called_once()
        loader.materialize_table_graph.assert_called_once()

    def test_llm_stage_error_returns_failed(self):
        work_item = TableWorkItem("cat", "sch", "tbl", "unity://cat.sch.tbl")

        connector = MagicMock()
        connector.extract_table.return_value = _make_extraction_assertions()

        llm_client = MagicMock(spec=LLMClient)
        llm_client.invoke.side_effect = LLMStageError(
            table_ref="unity://cat.sch.tbl",
            stage_name="L2 semantic",
            step_errors=[("plain_invoke", ValueError("fail"))],
        )

        loader = MagicMock()

        result = process_table(
            work_item, connector, llm_client, loader,
            run_id="run-1",
        )

        assert result.status == "failed"
        assert result.failed_stage == "L2 semantic"
        # No assertions committed
        loader.commit_table_assertions.assert_not_called()
        loader.materialize_table_graph.assert_not_called()

    def test_all_or_nothing_on_failure(self):
        """L1 extraction succeeds but L2 fails → nothing committed."""
        work_item = TableWorkItem("cat", "sch", "tbl", "unity://cat.sch.tbl")

        connector = MagicMock()
        connector.extract_table.return_value = _make_extraction_assertions()

        llm_client = MagicMock(spec=LLMClient)
        llm_client.invoke.side_effect = LLMStageError(
            table_ref="ref", stage_name="L2 semantic",
            step_errors=[("fail", ValueError("x"))],
        )

        loader = MagicMock()

        result = process_table(
            work_item, connector, llm_client, loader,
            run_id="run-1",
        )

        assert result.status == "failed"
        # Extraction produced assertions but they should NOT be committed
        loader.commit_table_assertions.assert_not_called()


# ---------------------------------------------------------------------------
# Resource factory tests (Task 8.4)
# ---------------------------------------------------------------------------

class TestResourceFactories:
    def test_connector_factory_creates_independent_instances(self):
        with patch(
            "sema.connectors.databricks.sql_connect"
        ) as mock_connect:
            mock_connect.return_value = MagicMock()
            from sema.models.config import (
                DatabricksConfig,
            )
            factory = DatabricksConnectorFactory(
                DatabricksConfig(
                    host="https://test.databricks.com",
                    token="dapi123",
                    http_path="/sql/1.0",
                ),
            )
            c1 = factory.create()
            c2 = factory.create()

            assert c1 is not c2
            assert mock_connect.call_count == 2

    def test_llm_client_factory_creates_independent_instances(self):
        call_count = [0]
        def make_llm():
            call_count[0] += 1
            return MagicMock()

        factory = LLMClientFactory(make_llm, retry_max_attempts=3)
        c1 = factory.create()
        c2 = factory.create()

        assert c1 is not c2
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# Assertion isolation tests (Task 8.5)
# ---------------------------------------------------------------------------

class TestAssertionIsolation:
    def test_separate_process_table_calls_independent(self):
        """Two process_table calls should not share assertion state."""
        results = []
        for i in range(2):
            work_item = TableWorkItem(
                "cat", "sch", f"tbl{i}", f"unity://cat.sch.tbl{i}"
            )
            connector = MagicMock()
            connector.extract_table.return_value = [
                _make_assertion(
                    f"unity://cat.sch.tbl{i}",
                    AssertionPredicate.TABLE_EXISTS,
                    {"table_type": "TABLE"},
                    run_id=f"run-{i}",
                ),
                _make_assertion(
                    f"unity://cat.sch.tbl{i}.col1",
                    AssertionPredicate.COLUMN_EXISTS,
                    {"data_type": "STRING", "nullable": True},
                    run_id=f"run-{i}",
                ),
            ]

            llm_client = MagicMock(spec=LLMClient)
            llm_client.invoke.return_value = TableInterpretation(
                entity_name=f"Entity{i}", properties=[]
            )

            loader = MagicMock()

            result = process_table(
                work_item, connector, llm_client, loader,
                run_id=f"run-{i}",
            )
            results.append(result)

            # Check committed assertions only reference this table
            committed = loader.commit_table_assertions.call_args[0][0]
            for a in committed:
                assert f"tbl{i}" in a.subject_ref

        assert all(r.status == "success" for r in results)


# ---------------------------------------------------------------------------
# Report aggregation tests (Task 8.6)
# ---------------------------------------------------------------------------

class TestReportAggregation:
    def test_aggregates_success_counts(self):
        results = [
            TableResult.success("t1", entities=2, properties=5, terms=3),
            TableResult.success("t2", entities=1, properties=3, terms=1),
        ]
        report = aggregate_report(results)
        assert report["tables_processed"] == 2
        assert report["entities_created"] == 3
        assert report["properties_created"] == 8
        assert report["terms_created"] == 4
        assert report["failed_tables"] == []

    def test_includes_failed_tables(self):
        results = [
            TableResult.success("t1", entities=1),
            TableResult.failed("t2", "L2 semantic", "LLM error"),
            TableResult.skipped("t3", "empty"),
        ]
        report = aggregate_report(results)
        assert report["tables_processed"] == 1
        assert report["tables_failed"] == 1
        assert report["tables_skipped"] == 1
        assert len(report["failed_tables"]) == 1
        assert report["failed_tables"][0]["table"] == "t2"
        assert report["failed_tables"][0]["stage"] == "L2 semantic"
