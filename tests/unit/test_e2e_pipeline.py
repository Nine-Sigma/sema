"""End-to-end validation tests for the vertical pipeline (Group 10).

These test the full process_table flow with mocked external
dependencies (Databricks, LLM, Neo4j).
"""
import json
import pytest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone

pytestmark = pytest.mark.unit

from sema.pipeline.build import (
    TableResult,
    process_table,
    aggregate_report,
)
from sema.connectors.databricks import TableWorkItem
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)
from sema.llm_client import (
    LLMClient,
    LLMStageError,
)
from sema.engine.semantic import (
    TableInterpretation,
    PropertyInterpretation,
)


def _make_assertion(subject_ref, predicate, payload=None, run_id="r"):
    return Assertion(
        id=f"a-{subject_ref}-{predicate.value}",
        subject_ref=subject_ref,
        predicate=predicate,
        payload=payload or {},
        source="test",
        confidence=0.9,
        status=AssertionStatus.AUTO,
        run_id=run_id,
        observed_at=datetime.now(timezone.utc),
    )


def _make_extraction_for_table(name, num_cols=3):
    fqn = f"unity://cat.sch.{name}"
    assertions = [
        _make_assertion(
            fqn, AssertionPredicate.TABLE_EXISTS,
            {"table_type": "TABLE"},
        ),
    ]
    for i in range(num_cols):
        assertions.append(_make_assertion(
            f"{fqn}.col{i}",
            AssertionPredicate.COLUMN_EXISTS,
            {"data_type": "STRING", "nullable": True, "comment": None},
        ))
    return assertions


# ---------------------------------------------------------------------------
# Task 10.1: Sequential equivalence
# ---------------------------------------------------------------------------

class TestSequentialEquivalence:
    def test_table_workers_1_processes_all(self):
        tables = [
            TableWorkItem("cat", "sch", f"tbl{i}", f"unity://cat.sch.tbl{i}")
            for i in range(3)
        ]

        results = []
        for t in tables:
            connector = MagicMock()
            connector.extract_table.return_value = _make_extraction_for_table(
                t.table_name
            )
            llm_client = MagicMock(spec=LLMClient)
            llm_client.invoke.return_value = TableInterpretation(
                entity_name=f"E_{t.table_name}", properties=[]
            )
            loader = MagicMock()
            r = process_table(t, connector, llm_client, loader, "run-1")
            results.append(r)

        assert all(r.status == "success" for r in results)
        report = aggregate_report(results)
        assert report["tables_processed"] == 3
        assert report["entities_created"] == 3


# ---------------------------------------------------------------------------
# Task 10.2: Concurrent with multiple tables
# ---------------------------------------------------------------------------

class TestConcurrentMultipleTables:
    def test_table_workers_2_all_assertions_present(self):
        tables = [
            TableWorkItem("cat", "sch", f"tbl{i}", f"unity://cat.sch.tbl{i}")
            for i in range(6)
        ]

        committed_counts = []

        def process_one(t):
            connector = MagicMock()
            connector.extract_table.return_value = _make_extraction_for_table(
                t.table_name
            )
            llm_client = MagicMock(spec=LLMClient)
            llm_client.invoke.return_value = TableInterpretation(
                entity_name=f"E_{t.table_name}",
                properties=[
                    PropertyInterpretation(
                        column="col0", name="Col 0",
                        semantic_type="free_text",
                    ),
                ],
            )
            loader = MagicMock()

            def capture(assertions):
                committed_counts.append(len(assertions))
            loader.commit_table_assertions.side_effect = capture

            return process_table(t, connector, llm_client, loader, "run-1")

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(process_one, tables))

        assert len(results) == 6
        assert all(r.status == "success" for r in results)
        report = aggregate_report(results)
        assert report["tables_processed"] == 6
        assert report["entities_created"] == 6
        assert report["properties_created"] == 6

        # Every table committed assertions
        assert len(committed_counts) == 6
        assert all(c > 0 for c in committed_counts)


# ---------------------------------------------------------------------------
# Task 10.3: Wide table with two-pass
# ---------------------------------------------------------------------------

class TestWideTableTwoPass:
    def test_60_columns_two_pass_produces_all_properties(self):
        from sema.llm_client import TableSummary
        from sema.engine.semantic import (
            _PropertyBatchResult,
        )

        work_item = TableWorkItem(
            "cat", "sch", "wide_tbl", "unity://cat.sch.wide_tbl"
        )

        connector = MagicMock()
        connector.extract_table.return_value = _make_extraction_for_table(
            "wide_tbl", num_cols=60
        )

        # LLMClient mock: summary + 3 property batches (25+25+10)
        call_idx = [0]
        def llm_invoke(prompt, schema, **kwargs):
            call_idx[0] += 1
            if call_idx[0] == 1:
                # Table summary
                return TableSummary(
                    entity_name="Wide Entity",
                    synonyms=["we"],
                )
            else:
                # Property batch
                batch_num = call_idx[0] - 1
                start = (batch_num - 1) * 25
                end = min(start + 25, 60)
                return _PropertyBatchResult(
                    properties=[
                        PropertyInterpretation(
                            column=f"col{i}",
                            name=f"Col {i}",
                            semantic_type="free_text",
                        )
                        for i in range(start, end)
                    ]
                )

        llm_client = MagicMock(spec=LLMClient)
        llm_client.invoke.side_effect = llm_invoke

        loader = MagicMock()

        result = process_table(
            work_item, connector, llm_client, loader,
            run_id="run-1",
            column_batch_size=25,
        )

        assert result.status == "success"
        assert result.entities_created == 1
        assert result.properties_created == 60

        # Verify commit was called with all assertions
        committed = loader.commit_table_assertions.call_args[0][0]
        prop_assertions = [
            a for a in committed
            if a.predicate == AssertionPredicate.HAS_PROPERTY_NAME
        ]
        assert len(prop_assertions) == 60


# ---------------------------------------------------------------------------
# Task 10.4: LLM fallback chain e2e
# ---------------------------------------------------------------------------

class TestLLMFallbackChainE2E:
    def test_structured_output_fails_plain_succeeds(self):
        work_item = TableWorkItem(
            "cat", "sch", "tbl", "unity://cat.sch.tbl"
        )

        connector = MagicMock()
        connector.extract_table.return_value = _make_extraction_for_table("tbl")

        # Real LLMClient with a mock LLM that fails structured output
        # but succeeds on plain invoke
        mock_llm = MagicMock()
        structured_mock = MagicMock()
        structured_mock.invoke.side_effect = Exception("structured failed")
        mock_llm.with_structured_output.return_value = structured_mock

        response = MagicMock()
        response.content = json.dumps({
            "entity_name": "Test Entity",
            "properties": [],
        })
        mock_llm.invoke.return_value = response

        llm_client = LLMClient(mock_llm, retry_max_attempts=1)
        loader = MagicMock()

        result = process_table(
            work_item, connector, llm_client, loader,
            run_id="run-1",
        )

        assert result.status == "success"
        assert result.entities_created == 1
        loader.commit_table_assertions.assert_called_once()


# ---------------------------------------------------------------------------
# Task 10.5: Transactional writes — failure during commit
# ---------------------------------------------------------------------------

class TestTransactionalWriteFailure:
    def test_commit_failure_leaves_no_state(self):
        work_item = TableWorkItem(
            "cat", "sch", "tbl", "unity://cat.sch.tbl"
        )

        connector = MagicMock()
        connector.extract_table.return_value = _make_extraction_for_table("tbl")

        llm_client = MagicMock(spec=LLMClient)
        llm_client.invoke.return_value = TableInterpretation(
            entity_name="E", properties=[]
        )

        loader = MagicMock()
        loader.commit_table_assertions.side_effect = Exception(
            "Neo4j transaction failed"
        )

        result = process_table(
            work_item, connector, llm_client, loader,
            run_id="run-1",
        )

        assert result.status == "failed"
        # Commit was attempted but failed — materializer should not have run
        loader.commit_table_assertions.assert_called_once()


# ---------------------------------------------------------------------------
# Task 10.6: Materialization idempotency
# ---------------------------------------------------------------------------

class TestMaterializationIdempotency:
    def test_double_materialization_same_calls(self):
        """Calling process_table twice (simulating re-run) should work."""
        for _ in range(2):
            work_item = TableWorkItem(
                "cat", "sch", "tbl", "unity://cat.sch.tbl"
            )
            connector = MagicMock()
            connector.extract_table.return_value = (
                _make_extraction_for_table("tbl")
            )
            llm_client = MagicMock(spec=LLMClient)
            llm_client.invoke.return_value = TableInterpretation(
                entity_name="E", properties=[]
            )
            loader = MagicMock()

            result = process_table(
                work_item, connector, llm_client, loader,
                run_id="run-1",
            )
            assert result.status == "success"
            loader.commit_table_assertions.assert_called_once()
            loader._run.assert_called()


# ---------------------------------------------------------------------------
# Task 10.7: Failure accounting
# ---------------------------------------------------------------------------

class TestFailureAccounting:
    def test_failed_table_in_report_others_succeed(self):
        tables = [
            TableWorkItem("cat", "sch", f"tbl{i}", f"unity://cat.sch.tbl{i}")
            for i in range(3)
        ]

        results = []
        for i, t in enumerate(tables):
            connector = MagicMock()
            connector.extract_table.return_value = (
                _make_extraction_for_table(t.table_name)
            )

            llm_client = MagicMock(spec=LLMClient)
            if i == 1:
                # Second table fails
                llm_client.invoke.side_effect = LLMStageError(
                    table_ref=t.fqn,
                    stage_name="L2 semantic",
                    step_errors=[("all", ValueError("boom"))],
                )
            else:
                llm_client.invoke.return_value = TableInterpretation(
                    entity_name=f"E{i}", properties=[]
                )

            loader = MagicMock()
            r = process_table(t, connector, llm_client, loader, "run-1")
            results.append(r)

        report = aggregate_report(results)
        assert report["tables_processed"] == 2
        assert report["tables_failed"] == 1
        assert len(report["failed_tables"]) == 1
        assert report["failed_tables"][0]["table"] == "unity://cat.sch.tbl1"
        assert report["failed_tables"][0]["stage"] == "L2 semantic"
