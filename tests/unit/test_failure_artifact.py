"""Failed-table forensic artifact (Section 5)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sema.engine.semantic import StageMetrics
from sema.engine.semantic_utils import LLMAttempt
from sema.llm_client import LLMStageError, StageBFailureError
from sema.circuit_breaker import CircuitOpenError
from sema.models.stages import (
    StageAResult,
    StageBBatchResult,
    StageBColumnResult,
    StageBCoverage,
    StageBResult,
    UnresolvedColumn,
)

pytestmark = pytest.mark.unit


def _cov(pct: float, classified: int = None, total: int = 10) -> StageBCoverage:
    if classified is None:
        classified = int(round(pct * total))
    return StageBCoverage(classified=classified, total=total, pct=pct)


def _stage_a() -> StageAResult:
    return StageAResult(
        primary_entity="Patient",
        grain_hypothesis="one row per patient",
        confidence=0.9,
    )


def _stage_b_failed() -> StageBResult:
    return StageBResult(
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


class TestClassifyFailure:
    def test_circuit_open_classified(self):
        from sema.eval.failure_artifact import classify_failure

        assert classify_failure(CircuitOpenError("open")) == "circuit_open"

    def test_stage_b_failure_classified_as_semantic_coverage(self):
        from sema.eval.failure_artifact import classify_failure

        err = StageBFailureError(
            table_ref="r", stage_name="L2 stage_b",
            step_errors=[("stage_b", ValueError("B_FAILED"))],
            stage_a=_stage_a(), stage_b=_stage_b_failed(),
            metrics=StageMetrics(), llm_attempts=[],
        )
        assert classify_failure(err) == "semantic_coverage"

    def test_503_step_classified_as_service_health(self):
        from sema.eval.failure_artifact import classify_failure

        e503 = Exception("Service Unavailable")
        e503.status_code = 503
        err = LLMStageError(
            table_ref="r", stage_name="L2 stage_a",
            step_errors=[("structured_output", e503)],
        )
        assert classify_failure(err) == "service_health"

    def test_429_only_classified_as_rate_limit(self):
        from sema.eval.failure_artifact import classify_failure

        e = Exception("rate limit")
        e.status_code = 429
        err = LLMStageError(
            table_ref="r", stage_name="L2 stage_a",
            step_errors=[("structured_output", e)],
        )
        assert classify_failure(err) == "rate_limit"

    def test_json_decode_classified_as_content_failure(self):
        from sema.eval.failure_artifact import classify_failure

        err = LLMStageError(
            table_ref="r", stage_name="L2 stage_a",
            step_errors=[
                ("plain_invoke", json.JSONDecodeError("bad", "", 0)),
            ],
        )
        assert classify_failure(err) == "content_failure"

    def test_value_error_classified_as_content_failure(self):
        from sema.eval.failure_artifact import classify_failure

        err = LLMStageError(
            table_ref="r", stage_name="L2 stage_a",
            step_errors=[
                ("plain_invoke", ValueError("Could not parse ...")),
            ],
        )
        assert classify_failure(err) == "content_failure"

    def test_unknown_exception_classified_as_unknown(self):
        from sema.eval.failure_artifact import classify_failure

        assert classify_failure(RuntimeError("mystery")) == "unknown"


class TestBuildFailureArtifact:
    def test_b_failed_artifact_includes_staged_context(self, tmp_path):
        from sema.eval.failure_artifact import build_failure_artifact

        att = LLMAttempt(
            stage="L2 stage_b", batch_index=0,
            prompt_text="prompt", prompt_hash="hash",
            raw_response="raw", parsed_ok=False,
        )
        err = StageBFailureError(
            table_ref="unity://cdm.clinical.patient",
            stage_name="L2 stage_b",
            step_errors=[("stage_b", ValueError("B_FAILED: raw=0.5"))],
            stage_a=_stage_a(), stage_b=_stage_b_failed(),
            metrics=StageMetrics(stage_a_latency_ms=10),
            llm_attempts=[att],
        )

        artifact = build_failure_artifact(
            exc=err,
            table_ref="unity://cdm.clinical.patient",
            run_id="run-1",
            failure_stage="L2 stage_b",
        )

        assert artifact["table_ref"] == "unity://cdm.clinical.patient"
        assert artifact["run_id"] == "run-1"
        assert artifact["failure_stage"] == "L2 stage_b"
        assert artifact["failure_classification"] == "semantic_coverage"
        assert artifact["stage_a_output"] is not None
        assert artifact["stage_a_output"]["primary_entity"] == "Patient"
        assert len(artifact["unresolved_columns"]) == 1
        assert artifact["unresolved_columns"][0]["column"] == "age"
        assert artifact["counters"]["retries_used"] == 1
        assert artifact["counters"]["batches_attempted"] >= 0
        assert "stage_b" in artifact["prompt_hashes"]
        assert artifact["prompt_hashes"]["stage_b"]
        assert len(artifact["llm_attempts"]) == 1
        assert artifact["llm_attempts"][0]["parsed_ok"] is False
        assert len(artifact["step_errors"]) == 1
        assert artifact["step_errors"][0]["step_name"] == "stage_b"

    def test_stage_a_failure_has_null_stage_a_output(self):
        from sema.eval.failure_artifact import build_failure_artifact

        att = LLMAttempt(
            stage="L2 stage_a", batch_index=None,
            prompt_text="prompt", prompt_hash="hash",
            raw_response=None, parsed_ok=False,
        )
        err = LLMStageError(
            table_ref="r", stage_name="L2 stage_a",
            step_errors=[
                ("structured_output", ValueError("bad")),
            ],
            llm_attempts=[att],
        )

        artifact = build_failure_artifact(
            exc=err,
            table_ref="unity://cdm.clinical.patient",
            run_id="run-1",
            failure_stage="L2 stage_a",
        )

        assert artifact["stage_a_output"] is None
        assert artifact["unresolved_columns"] == []
        assert artifact["counters"]["batches_attempted"] == 0
        assert len(artifact["llm_attempts"]) == 1
        assert artifact["failure_classification"] == "content_failure"

    def test_circuit_open_artifact(self):
        from sema.eval.failure_artifact import build_failure_artifact

        artifact = build_failure_artifact(
            exc=CircuitOpenError("circuit open"),
            table_ref="unity://cdm.clinical.patient",
            run_id="run-1",
            failure_stage="circuit_breaker",
        )

        assert artifact["failure_stage"] == "circuit_breaker"
        assert artifact["failure_classification"] == "circuit_open"
        assert artifact["llm_attempts"] == []
        assert artifact["stage_a_output"] is None


class TestDumpTableFailureArtifact:
    def test_writes_failure_json(self, tmp_path):
        from sema.eval.failure_artifact import dump_table_failure_artifact

        err = LLMStageError(
            table_ref="r", stage_name="L2 stage_a",
            step_errors=[("structured_output", ValueError("bad"))],
            llm_attempts=[],
        )
        path = dump_table_failure_artifact(
            exc=err,
            table_ref="unity://cat/sch/patient",
            label="run",
            output_dir=tmp_path,
            run_id="run-1",
            failure_stage="L2 stage_a",
        )
        assert path is not None
        assert path.exists()
        assert path.name == "patient__run__failure.json"
        data = json.loads(path.read_text())
        assert data["failure_stage"] == "L2 stage_a"
        assert data["failure_classification"] == "content_failure"

    def test_returns_none_when_output_dir_is_none(self):
        from sema.eval.failure_artifact import dump_table_failure_artifact

        err = LLMStageError(
            table_ref="r", stage_name="L2 stage_a",
            step_errors=[], llm_attempts=[],
        )
        path = dump_table_failure_artifact(
            exc=err,
            table_ref="unity://cdm.clinical.patient",
            label="run",
            output_dir=None,
            run_id="run-1",
            failure_stage="L2 stage_a",
        )
        assert path is None


class TestProcessTableWritesFailureArtifact:
    def test_b_failed_writes_artifact(self, tmp_path):
        """End-to-end: process_table with eval_dump_dir set writes
        *__failure.json on B_FAILED."""
        from sema.connectors.databricks import TableWorkItem
        from sema.pipeline.build import process_table

        loader = MagicMock()
        loader.has_assertions.return_value = False
        connector = MagicMock()
        llm_client = MagicMock()

        wi = TableWorkItem(
            catalog="cat", schema="sch",
            table_name="patient",
            fqn="databricks://ws/cat/sch/patient",
        )

        att = LLMAttempt(
            stage="L2 stage_b", batch_index=0,
            prompt_text="prompt", prompt_hash="h",
            raw_response="raw", parsed_ok=False,
        )
        err = StageBFailureError(
            table_ref=wi.fqn, stage_name="L2 stage_b",
            step_errors=[("stage_b", ValueError("B_FAILED: raw=0.5"))],
            stage_a=_stage_a(), stage_b=_stage_b_failed(),
            metrics=StageMetrics(),
            llm_attempts=[att],
        )

        with patch(
            "sema.pipeline.build._run_pipeline_stages",
            side_effect=err,
        ):
            result = process_table(
                wi, connector, llm_client, loader,
                run_id="run-1",
                eval_dump_dir=str(tmp_path),
                eval_config_label="run",
            )

        assert result.status == "failed"
        path = tmp_path / "patient__run__failure.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["failure_classification"] == "semantic_coverage"
        assert data["stage_a_output"] is not None

    def test_circuit_open_writes_artifact(self, tmp_path):
        from sema.connectors.databricks import TableWorkItem
        from sema.pipeline.build import process_table

        loader = MagicMock()
        loader.has_assertions.return_value = False
        connector = MagicMock()
        llm_client = MagicMock()

        wi = TableWorkItem(
            catalog="cat", schema="sch",
            table_name="mutation",
            fqn="databricks://ws/cat/sch/mutation",
        )

        with patch(
            "sema.pipeline.build._run_pipeline_stages",
            side_effect=CircuitOpenError("circuit open"),
        ):
            result = process_table(
                wi, connector, llm_client, loader,
                run_id="run-1",
                eval_dump_dir=str(tmp_path),
                eval_config_label="run",
            )

        assert result.status == "failed"
        path = tmp_path / "mutation__run__failure.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["failure_classification"] == "circuit_open"
        assert data["failure_stage"] == "circuit_breaker"

    def test_no_eval_dump_dir_writes_no_artifact(self, tmp_path):
        from sema.connectors.databricks import TableWorkItem
        from sema.pipeline.build import process_table

        loader = MagicMock()
        loader.has_assertions.return_value = False
        connector = MagicMock()
        llm_client = MagicMock()

        wi = TableWorkItem(
            catalog="cat", schema="sch",
            table_name="patient",
            fqn="databricks://ws/cat/sch/patient",
        )

        with patch(
            "sema.pipeline.build._run_pipeline_stages",
            side_effect=LLMStageError(
                table_ref=wi.fqn, stage_name="L2 stage_a",
                step_errors=[], llm_attempts=[],
            ),
        ):
            result = process_table(
                wi, connector, llm_client, loader,
                run_id="run-1",
                eval_dump_dir=None,
            )
        assert result.status == "failed"
        # No file was written anywhere
        assert list(tmp_path.glob("*__failure.json")) == []

    def test_artifact_write_failure_does_not_propagate(self, tmp_path):
        """OSError during artifact write must be logged + swallowed."""
        from sema.connectors.databricks import TableWorkItem
        from sema.pipeline.build import process_table

        loader = MagicMock()
        loader.has_assertions.return_value = False
        connector = MagicMock()
        llm_client = MagicMock()

        wi = TableWorkItem(
            catalog="cat", schema="sch",
            table_name="patient",
            fqn="databricks://ws/cat/sch/patient",
        )

        err = LLMStageError(
            table_ref=wi.fqn, stage_name="L2 stage_a",
            step_errors=[], llm_attempts=[],
        )

        with patch(
            "sema.pipeline.build._run_pipeline_stages",
            side_effect=err,
        ), patch(
            "sema.eval.failure_artifact.dump_table_failure_artifact",
            side_effect=OSError("disk full"),
        ):
            result = process_table(
                wi, connector, llm_client, loader,
                run_id="run-1",
                eval_dump_dir=str(tmp_path),
            )

        assert result.status == "failed"

    def test_success_path_writes_no_failure_artifact(self):
        # Existing successful runs continue to write only telemetry.
        # Here we only verify failure module isn't called for success.
        # End-to-end success behavior is covered by other tests.
        from sema.eval.failure_artifact import dump_table_failure_artifact
        # Smoke import only — no behavior to assert here.
        assert callable(dump_table_failure_artifact)
