from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from sema.connectors.databricks import TableWorkItem
from sema.engine.semantic import (
    PropertyInterpretation,
    TableInterpretation,
)
from sema.llm_client import LLMClient
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)
from sema.pipeline.build import TableResult, process_table

pytestmark = pytest.mark.unit


def _make_assertion(
    subject_ref: str,
    predicate: AssertionPredicate,
    payload: dict | None = None,
    source: str = "test",
    run_id: str = "run-1",
) -> Assertion:
    return Assertion(
        id=f"a-{subject_ref}-{predicate.value}",
        subject_ref=subject_ref,
        predicate=predicate,
        payload=payload or {},
        source=source,
        confidence=0.9,
        status=AssertionStatus.AUTO,
        run_id=run_id,
        observed_at=datetime.now(timezone.utc),
    )


def _make_extraction_assertions() -> list[Assertion]:
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


def _make_stored_assertion_dicts() -> list[dict]:
    return [
        {
            "id": "a-1",
            "subject_ref": "unity://cat.sch.tbl",
            "predicate": "table_exists",
            "payload": '{"table_type": "TABLE"}',
            "object_ref": None,
            "source": "test",
            "confidence": 0.9,
            "status": "auto",
            "run_id": "old-run",
            "observed_at": "2025-01-01T00:00:00+00:00",
        },
        {
            "id": "a-2",
            "subject_ref": "unity://cat.sch.tbl.col1",
            "predicate": "column_exists",
            "payload": '{"data_type": "STRING", "nullable": true}',
            "object_ref": None,
            "source": "test",
            "confidence": 0.9,
            "status": "auto",
            "run_id": "old-run",
            "observed_at": "2025-01-01T00:00:00+00:00",
        },
    ]


def _build_work_item() -> TableWorkItem:
    return TableWorkItem("cat", "sch", "tbl", "unity://cat.sch.tbl")


class TestResumeBuild:
    def test_resume_skips_processed_table(self) -> None:
        work_item = _build_work_item()
        connector = MagicMock()
        llm_client = MagicMock(spec=LLMClient)
        loader = MagicMock()
        loader.has_assertions.return_value = True
        loader.load_assertions.return_value = _make_stored_assertion_dicts()

        result = process_table(
            work_item, connector, llm_client, loader,
            run_id="run-1", resume=True,
        )

        assert result.status == "skipped"
        assert result.skip_reason == "resume: assertions exist"
        connector.extract_table.assert_not_called()
        llm_client.invoke.assert_not_called()

    def test_resume_processes_new_table(self) -> None:
        work_item = _build_work_item()
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
        loader.has_assertions.return_value = False

        result = process_table(
            work_item, connector, llm_client, loader,
            run_id="run-1", resume=True,
        )

        assert result.status == "success"
        connector.extract_table.assert_called_once()
        llm_client.invoke.assert_called()

    def test_no_resume_processes_all(self) -> None:
        work_item = _build_work_item()
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
            run_id="run-1", resume=False,
        )

        assert result.status == "success"
        loader.has_assertions.assert_not_called()
        connector.extract_table.assert_called_once()

    def test_resume_preserves_existing_assertions(self) -> None:
        work_item = _build_work_item()
        connector = MagicMock()
        llm_client = MagicMock(spec=LLMClient)
        loader = MagicMock()
        loader.has_assertions.return_value = True
        loader.load_assertions.return_value = _make_stored_assertion_dicts()

        process_table(
            work_item, connector, llm_client, loader,
            run_id="run-1", resume=True,
        )

        loader.store_assertion.assert_not_called()
        loader.commit_table_assertions.assert_not_called()

    def test_resume_no_prefix_collision(self) -> None:
        work_item = _build_work_item()
        connector = MagicMock()
        llm_client = MagicMock(spec=LLMClient)
        loader = MagicMock()
        loader.has_assertions.return_value = True
        loader.load_assertions.return_value = _make_stored_assertion_dicts()

        process_table(
            work_item, connector, llm_client, loader,
            run_id="run-1", resume=True,
        )

        loader.has_assertions.assert_called_once_with("unity://cat.sch.tbl")

    def test_resume_rematerializes_skipped_tables(self) -> None:
        work_item = _build_work_item()
        connector = MagicMock()
        llm_client = MagicMock(spec=LLMClient)
        loader = MagicMock()
        loader.has_assertions.return_value = True
        loader.load_assertions.return_value = _make_stored_assertion_dicts()

        process_table(
            work_item, connector, llm_client, loader,
            run_id="run-1", resume=True,
        )

        loader.materialize_table_graph.assert_called_once()
