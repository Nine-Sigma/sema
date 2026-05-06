"""StageBFailureError carries staged context + llm_attempts (Section 4)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sema.engine.semantic import SemanticEngine, StageMetrics
from sema.engine.semantic_utils import LLMAttempt
from sema.llm_client import LLMClient, LLMStageError, StageBFailureError
from sema.models.stages import (
    StageAResult,
    StageBBatchResult,
    StageBColumnResult,
    StageBCoverage,
    StageBResult,
    UnresolvedColumn,
)


def _cov(pct: float = 1.0) -> StageBCoverage:
    return StageBCoverage(classified=int(pct * 10), total=10, pct=pct)

pytestmark = pytest.mark.unit


def _stage_a() -> StageAResult:
    return StageAResult(
        primary_entity="Patient",
        grain_hypothesis="one row per patient",
        confidence=0.9,
    )


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


class TestLLMStageErrorCarriesAttempts:
    def test_default_llm_attempts_empty(self):
        err = LLMStageError(
            table_ref="r",
            stage_name="L2 stage_a",
            step_errors=[("structured_output", ValueError("bad"))],
        )
        assert err.llm_attempts == []

    def test_can_attach_attempts(self):
        att = LLMAttempt(
            stage="L2 stage_a", batch_index=None,
            prompt_text="p", prompt_hash="h",
            raw_response=None, parsed_ok=False,
        )
        err = LLMStageError(
            table_ref="r",
            stage_name="L2 stage_a",
            step_errors=[("structured_output", ValueError("bad"))],
            llm_attempts=[att],
        )
        assert err.llm_attempts == [att]


class TestStageBFailureError:
    def test_is_a_llm_stage_error(self):
        sa = _stage_a()
        sb = StageBResult(
            status="B_FAILED",
            batch_results=[],
            raw_coverage=_cov(0.5),
            critical_coverage=_cov(1.0),
            unresolved_columns=[],
            retries_used=0,
            splits_used=0,
            rescues_used=0,
        )
        err = StageBFailureError(
            table_ref="r",
            stage_name="L2 stage_b",
            step_errors=[("stage_b", ValueError("B_FAILED: raw=0.50"))],
            stage_a=sa, stage_b=sb,
            metrics=StageMetrics(),
            llm_attempts=[],
        )
        assert isinstance(err, LLMStageError)

    def test_carries_staged_context(self):
        sa = _stage_a()
        sb = StageBResult(
            status="B_FAILED",
            batch_results=[StageBBatchResult(columns=[
                StageBColumnResult(
                    column="patient_id",
                    canonical_property_label="id",
                    semantic_type="identifier",
                    entity_role="primary_key",
                    needs_stage_c=False,
                ),
            ])],
            raw_coverage=_cov(0.5),
            critical_coverage=_cov(1.0),
            unresolved_columns=[
                UnresolvedColumn(
                    column="age", reason="execution_failure", tier="peripheral",
                ),
            ],
            retries_used=1,
            splits_used=0,
            rescues_used=0,
        )
        m = StageMetrics(stage_a_latency_ms=10, stage_b_latency_ms=20)
        att = LLMAttempt(
            stage="L2 stage_b", batch_index=0,
            prompt_text="p", prompt_hash="h",
            raw_response="r", parsed_ok=False,
        )
        err = StageBFailureError(
            table_ref="r",
            stage_name="L2 stage_b",
            step_errors=[("stage_b", ValueError("B_FAILED"))],
            stage_a=sa, stage_b=sb, metrics=m,
            llm_attempts=[att],
        )
        assert err.stage_a is sa
        assert err.stage_b is sb
        assert err.metrics is m
        assert err.llm_attempts == [att]

    def test_caught_by_existing_except_clause(self):
        sa = _stage_a()
        sb = StageBResult(
            status="B_FAILED", batch_results=[],
            raw_coverage=_cov(0.5), critical_coverage=_cov(1.0),
            unresolved_columns=[], retries_used=0,
            splits_used=0, rescues_used=0,
        )
        try:
            raise StageBFailureError(
                table_ref="r",
                stage_name="L2 stage_b",
                step_errors=[],
                stage_a=sa, stage_b=sb,
                metrics=StageMetrics(),
                llm_attempts=[],
            )
        except LLMStageError as caught:
            assert isinstance(caught, StageBFailureError)


class TestEngineRaisesStageBFailureError:
    def test_synthetic_b_failed_raises_stage_b_failure_error(self):
        client = MagicMock(spec=LLMClient)
        client.invoke.side_effect = [
            _stage_a(),
            StageBBatchResult(columns=[
                StageBColumnResult(
                    column="patient_id",
                    canonical_property_label="id",
                    semantic_type="identifier",
                    entity_role="primary_key",
                    needs_stage_c=False,
                ),
            ]),
            # Second batch returns no cols -> raw coverage drops -> B_FAILED
            StageBBatchResult(columns=[]),
        ]
        client.last_response = None
        engine = SemanticEngine(
            llm_client=client, run_id="r",
            capture_llm_attempts=True,
            column_batch_size=1,
        )
        with pytest.raises(StageBFailureError) as exc_info:
            engine.interpret_table_staged_with_metrics(_meta())

        err = exc_info.value
        assert err.stage_a is not None
        assert err.stage_b is not None
        assert err.stage_b.status == "B_FAILED"
        assert err.metrics is not None
        # attempts include stage_a + stage_b batches
        stages = [a.stage for a in err.llm_attempts]
        assert "L2 stage_a" in stages
        assert "L2 stage_b" in stages

    def test_stage_a_failure_carries_attempts_on_llm_stage_error(self):
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
        with pytest.raises(LLMStageError) as exc_info:
            engine.interpret_table_staged_with_metrics(_meta())

        err = exc_info.value
        # Engine attaches the attempt buffer at every raise site
        assert len(err.llm_attempts) >= 1
        assert err.llm_attempts[0].stage == "L2 stage_a"
        assert err.llm_attempts[0].parsed_ok is False
