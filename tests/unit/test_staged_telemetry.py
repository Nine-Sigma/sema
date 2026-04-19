"""Tests for per-stage latency and token aggregation."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sema.engine.semantic import SemanticEngine
from sema.llm_client import InvocationStats
from sema.models.stages import (
    StageAResult,
    StageBBatchResult,
    StageBColumnResult,
)

pytestmark = pytest.mark.unit


def _stage_a_result() -> StageAResult:
    return StageAResult(
        primary_entity="Patient",
        grain_hypothesis="patient-level",
        confidence=0.9,
    )


def _stage_b_batch(cols: list[str]) -> StageBBatchResult:
    return StageBBatchResult(
        columns=[
            StageBColumnResult(
                column=c,
                canonical_property_label=c.replace("_", " "),
                semantic_type="identifier",
                entity_role="primary_identifier",
                needs_stage_c=False,
            )
            for c in cols
        ],
    )


class TestStagedTelemetryCapture:
    def test_interpret_table_staged_accumulates_tokens(self) -> None:
        client = MagicMock()
        responses = iter([
            _stage_a_result(),
            _stage_b_batch(["id", "name"]),
        ])
        stats_sequence = iter([
            InvocationStats(prompt_tokens=100, completion_tokens=30),
            InvocationStats(prompt_tokens=400, completion_tokens=150),
        ])

        def _invoke(*args: object, **kwargs: object) -> object:
            client.last_stats = next(stats_sequence)
            return next(responses)

        client.invoke = _invoke
        client.last_stats = InvocationStats()

        engine = SemanticEngine(
            llm_client=client, column_batch_size=25,
        )
        table_meta = {
            "table_name": "patient",
            "table_ref": "databricks://x/c/s/patient",
            "columns": [
                {"name": "id", "data_type": "STRING"},
                {"name": "name", "data_type": "STRING"},
            ],
        }

        _, _, _, _, metrics = (
            engine.interpret_table_staged_with_metrics(table_meta)
        )
        assert metrics.tokens_input == 500
        assert metrics.tokens_output == 180
        assert metrics.stage_a_latency_ms >= 0
        assert metrics.stage_b_latency_ms >= 0
        assert metrics.stage_c_calls == 0
