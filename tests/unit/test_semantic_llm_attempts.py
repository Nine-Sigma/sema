"""Engine-local LLM-attempt buffer (Section 3 of stage-b-failure-containment).

Captures forensics for every LLMClient.invoke call during a single
interpret_table_* execution. Opt-in via the engine's capture flag.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sema.engine.semantic import SemanticEngine
from sema.llm_client import LLMClient, LLMStageError
from sema.models.stages import (
    StageAResult,
    StageBBatchResult,
    StageBColumnResult,
)

pytestmark = pytest.mark.unit


def _meta() -> dict:
    return {
        "table_ref": "unity://cdm.clinical.patient",
        "table_name": "patient",
        "columns": [
            {"name": "patient_id", "data_type": "STRING"},
            {"name": "age", "data_type": "INTEGER"},
        ],
        "sample_rows": [],
        "comment": None,
    }


def _stage_a() -> StageAResult:
    return StageAResult(
        primary_entity="Patient",
        grain_hypothesis="one row per patient",
        confidence=0.9,
    )


def _stage_b_one_col(col: str) -> StageBBatchResult:
    return StageBBatchResult(columns=[
        StageBColumnResult(
            column=col,
            canonical_property_label=col,
            semantic_type="identifier",
            entity_role="primary_key",
            needs_stage_c=False,
        ),
    ])


class TestLLMAttemptModel:
    def test_attempt_dataclass_fields(self):
        from sema.engine.semantic_utils import LLMAttempt

        a = LLMAttempt(
            stage="L2 stage_a",
            batch_index=None,
            prompt_text="prompt",
            prompt_hash="abc",
            raw_response="raw",
            parsed_ok=True,
        )
        assert a.stage == "L2 stage_a"
        assert a.batch_index is None
        assert a.parsed_ok is True


class TestEngineCaptureFlag:
    def test_buffer_empty_when_capture_disabled(self):
        client = MagicMock(spec=LLMClient)
        client.invoke.side_effect = [
            _stage_a(),
            _stage_b_one_col("patient_id"),
            _stage_b_one_col("age"),
        ]
        client.last_response = None
        engine = SemanticEngine(
            llm_client=client, run_id="r",
            capture_llm_attempts=False,
            column_batch_size=1,
        )
        engine.interpret_table(_meta())
        assert engine.snapshot_llm_attempts() == []

    def test_buffer_populated_for_successful_invocations(self):
        client = MagicMock(spec=LLMClient)
        client.invoke.side_effect = [
            _stage_a(),
            _stage_b_one_col("patient_id"),
            _stage_b_one_col("age"),
        ]
        client.last_response = "raw text"
        engine = SemanticEngine(
            llm_client=client, run_id="r",
            capture_llm_attempts=True,
            column_batch_size=1,
        )
        engine.interpret_table(_meta())
        snap = engine.snapshot_llm_attempts()
        stages = [a.stage for a in snap]
        assert "L2 stage_a" in stages
        assert stages.count("L2 stage_b") >= 2
        assert all(a.parsed_ok for a in snap)
        assert all(a.prompt_text and a.prompt_hash for a in snap)

    def test_buffer_records_failed_invocation(self):
        """Failed invocation records parsed_ok=False with prompt + hash."""
        client = MagicMock(spec=LLMClient)
        client.invoke.side_effect = LLMStageError(
            table_ref="r", stage_name="L2 stage_a",
            step_errors=[("structured_output", ValueError("bad"))],
        )
        client.last_response = None
        engine = SemanticEngine(
            llm_client=client, run_id="r",
            capture_llm_attempts=True,
        )
        with pytest.raises(LLMStageError):
            engine.interpret_table(_meta())
        snap = engine.snapshot_llm_attempts()
        assert len(snap) >= 1
        assert snap[-1].parsed_ok is False
        assert snap[-1].stage == "L2 stage_a"
        assert snap[-1].prompt_text

    def test_buffer_reset_between_tables(self):
        client = MagicMock(spec=LLMClient)
        client.invoke.side_effect = [
            _stage_a(), _stage_b_one_col("patient_id"), _stage_b_one_col("age"),
            _stage_a(), _stage_b_one_col("patient_id"), _stage_b_one_col("age"),
        ]
        client.last_response = "raw"
        engine = SemanticEngine(
            llm_client=client, run_id="r",
            capture_llm_attempts=True,
            column_batch_size=1,
        )
        engine.interpret_table(_meta())
        first_count = len(engine.snapshot_llm_attempts())
        assert first_count > 0

        engine.interpret_table(_meta())
        second_count = len(engine.snapshot_llm_attempts())
        # Each table populates from scratch; second table buffer reflects
        # only its own invocations (not first + second).
        assert second_count == first_count


class TestStageBBatchIndexing:
    def test_stage_b_attempts_have_batch_index(self):
        client = MagicMock(spec=LLMClient)
        client.invoke.side_effect = [
            _stage_a(),
            _stage_b_one_col("patient_id"),
            _stage_b_one_col("age"),
        ]
        client.last_response = "raw"
        engine = SemanticEngine(
            llm_client=client, run_id="r",
            capture_llm_attempts=True,
            column_batch_size=1,
        )
        engine.interpret_table(_meta())
        snap = engine.snapshot_llm_attempts()
        b_attempts = [a for a in snap if a.stage == "L2 stage_b"]
        # column_batch_size=1, two cols -> two batches with indices 0, 1
        indices = [a.batch_index for a in b_attempts]
        assert sorted(i for i in indices if i is not None) == [0, 1]


class TestLLMClientLastResponseProperty:
    def test_last_response_property_exposed(self):
        from sema.llm_client import LLMClient

        llm = MagicMock(spec=["invoke"])
        response = MagicMock()
        response.content = '{"vocabulary": "ICD-10"}'
        llm.invoke.return_value = response

        from sema.llm_client import VocabularyDetection
        client = LLMClient(llm, retry_max_attempts=1)
        client.invoke("p", VocabularyDetection)
        # last_response should expose the raw model response
        assert client.last_response is response
