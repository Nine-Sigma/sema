"""Tests for table-level parallelism (Group 9)."""
import time
import pytest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch
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
from sema.llm_client import LLMClient
from sema.models.stages import (
    StageAResult,
    StageBBatchResult,
    StageBColumnResult,
)


def _make_assertion(subject_ref, predicate, payload=None, run_id="r"):
    return Assertion(
        id=f"a-{subject_ref}",
        subject_ref=subject_ref,
        predicate=predicate,
        payload=payload or {},
        source="test",
        confidence=0.9,
        status=AssertionStatus.AUTO,
        run_id=run_id,
        observed_at=datetime.now(timezone.utc),
    )


def _make_mock_resources(table_name):
    """Create mocked connector, llm_client, loader for a table."""
    connector = MagicMock()
    connector.extract_table.return_value = [
        _make_assertion(
            f"unity://cat.sch.{table_name}",
            AssertionPredicate.TABLE_EXISTS,
            {"table_type": "TABLE"},
        ),
        _make_assertion(
            f"unity://cat.sch.{table_name}.col1",
            AssertionPredicate.COLUMN_EXISTS,
            {"data_type": "STRING", "nullable": True},
        ),
    ]

    llm_client = MagicMock(spec=LLMClient)
    llm_client.invoke.side_effect = [
        StageAResult(
            primary_entity=f"Entity_{table_name}",
            grain_hypothesis="one row per entity",
            confidence=0.9,
        ),
        StageBBatchResult(columns=[
            StageBColumnResult(
                column="col1",
                canonical_property_label="column one",
                semantic_type="identifier",
                entity_role="attribute",
                needs_stage_c=False,
            ),
        ]),
    ]

    loader = MagicMock()
    return connector, llm_client, loader


# ---------------------------------------------------------------------------
# Sequential mode tests (Task 9.1)
# ---------------------------------------------------------------------------

class TestSequentialMode:
    def test_single_worker_processes_all_tables(self):
        results = []
        for i in range(5):
            name = f"tbl{i}"
            work_item = TableWorkItem(
                "cat", "sch", name, f"unity://cat.sch.{name}"
            )
            connector, llm_client, loader = _make_mock_resources(name)
            r = process_table(
                work_item, connector, llm_client, loader,
                run_id="run-1",
            )
            results.append(r)

        assert all(r.status == "success" for r in results)
        report = aggregate_report(results)
        assert report["tables_processed"] == 5


# ---------------------------------------------------------------------------
# Concurrent mode tests (Task 9.2)
# ---------------------------------------------------------------------------

class TestConcurrentMode:
    def test_parallel_produces_same_results(self):
        """table_workers=4 with 20 tables produces same results."""
        tables = [
            TableWorkItem(
                "cat", "sch", f"tbl{i}", f"unity://cat.sch.tbl{i}"
            )
            for i in range(20)
        ]

        def process_one(work_item):
            connector, llm_client, loader = _make_mock_resources(
                work_item.table_name
            )
            return process_table(
                work_item, connector, llm_client, loader,
                run_id="run-1",
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_one, tables))

        assert len(results) == 20
        assert all(r.status == "success" for r in results)
        report = aggregate_report(results)
        assert report["tables_processed"] == 20
        assert report["entities_created"] == 20


# ---------------------------------------------------------------------------
# Worker resource isolation tests (Task 9.3)
# ---------------------------------------------------------------------------

class TestWorkerResourceIsolation:
    def test_each_worker_gets_own_resources(self):
        """Verify no shared state between concurrent workers."""
        tables = [
            TableWorkItem(
                "cat", "sch", f"tbl{i}", f"unity://cat.sch.tbl{i}"
            )
            for i in range(4)
        ]

        committed_refs = []

        def process_one(work_item):
            connector, llm_client, loader = _make_mock_resources(
                work_item.table_name
            )
            # Track what each loader commits
            def capture_commit(assertions):
                refs = [a.subject_ref for a in assertions]
                committed_refs.append(
                    (work_item.table_name, refs)
                )
            loader.commit_table_assertions.side_effect = capture_commit

            return process_table(
                work_item, connector, llm_client, loader,
                run_id="run-1",
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_one, tables))

        assert all(r.status == "success" for r in results)

        # Each commit should only reference its own table
        for table_name, refs in committed_refs:
            for ref in refs:
                assert table_name in ref


# ---------------------------------------------------------------------------
# No-deadlock tests (Task 9.4)
# ---------------------------------------------------------------------------

class TestNoDeadlock:
    def test_4_workers_4_tables_completes(self):
        """With all inner work sequential, the pipeline should complete
        without hanging even at full utilization."""
        tables = [
            TableWorkItem(
                "cat", "sch", f"tbl{i}", f"unity://cat.sch.tbl{i}"
            )
            for i in range(4)
        ]

        def process_one(work_item):
            connector, llm_client, loader = _make_mock_resources(
                work_item.table_name
            )
            # Simulate some work
            time.sleep(0.01)
            return process_table(
                work_item, connector, llm_client, loader,
                run_id="run-1",
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Should complete within 5 seconds (not deadlock)
            futures = [
                executor.submit(process_one, t) for t in tables
            ]
            results = [f.result(timeout=5) for f in futures]

        assert len(results) == 4
        assert all(r.status == "success" for r in results)
