"""Tests for Stage C: conditional value interpretation (tasks 9.2–9.7)."""
import pytest
from unittest.mock import MagicMock

from sema.models.stages import (
    StageAResult,
    StageBBatchResult,
    StageBColumnResult,
    StageBCoverage,
    StageBResult,
    StageCBatchResult,
    StageCResult,
    UnresolvedColumn,
)

pytestmark = pytest.mark.unit


def _stage_a() -> StageAResult:
    return StageAResult(
        primary_entity="Patient",
        grain_hypothesis="one row per patient",
        confidence=0.9,
    )


def _b_col(
    name: str,
    sem_type: str = "categorical",
    needs_c: bool = False,
) -> StageBColumnResult:
    return StageBColumnResult(
        column=name,
        canonical_property_label=name,
        semantic_type=sem_type,
        confidence=0.9,
        needs_stage_c=needs_c,
    )


def _stage_b(columns: list[StageBColumnResult]) -> StageBResult:
    batch = StageBBatchResult(columns=columns)
    return StageBResult(
        status="B_SUCCESS",
        batch_results=[batch],
        raw_coverage=StageBCoverage(
            classified=len(columns),
            total=len(columns),
            pct=1.0,
        ),
        critical_coverage=StageBCoverage(
            classified=len(columns),
            total=len(columns),
            pct=1.0,
        ),
    )


class TestStageCTrigger:
    """Task 9.2: deterministic trigger function."""

    def test_flagged_column_triggers(self) -> None:
        from sema.engine.stage_utils import should_trigger_stage_c

        col = _b_col("gender", "categorical", needs_c=True)
        assert should_trigger_stage_c(col) is True

    def test_identifier_excluded(self) -> None:
        from sema.engine.stage_utils import should_trigger_stage_c

        col = _b_col("patient_id", "identifier", needs_c=True)
        assert should_trigger_stage_c(col) is False

    def test_temporal_excluded(self) -> None:
        from sema.engine.stage_utils import should_trigger_stage_c

        col = _b_col("start_date", "temporal", needs_c=True)
        assert should_trigger_stage_c(col) is False

    def test_free_text_excluded(self) -> None:
        from sema.engine.stage_utils import should_trigger_stage_c

        col = _b_col("notes", "free_text", needs_c=True)
        assert should_trigger_stage_c(col) is False

    def test_not_flagged_skipped(self) -> None:
        from sema.engine.stage_utils import should_trigger_stage_c

        col = _b_col("cancer_type", "categorical", needs_c=False)
        assert should_trigger_stage_c(col) is False

    def test_patient_identifier_type_excluded(self) -> None:
        from sema.engine.stage_utils import should_trigger_stage_c

        col = _b_col("pid", "patient identifier", needs_c=True)
        assert should_trigger_stage_c(col) is False


class TestStageCPrompt:
    """Task 9.3: Stage C prompt construction."""

    def test_prompt_includes_column_and_values(self) -> None:
        from sema.engine.stage_utils import build_stage_c_prompt

        prompt = build_stage_c_prompt(
            columns_with_values=[
                {"column": "gender", "values": ["Male (55%)", "Female (43%)", "Other (2%)"]},
            ],
            stage_a=_stage_a(),
            domain_context=None,
        )
        assert "gender" in prompt
        assert "Male" in prompt
        assert "decoded_categories" in prompt

    def test_prompt_includes_entity_context(self) -> None:
        from sema.engine.stage_utils import build_stage_c_prompt

        prompt = build_stage_c_prompt(
            columns_with_values=[
                {"column": "os_status", "values": ["0:LIVING", "1:DECEASED"]},
            ],
            stage_a=_stage_a(),
            domain_context=None,
        )
        assert "Patient" in prompt

    def test_prompt_batches_multiple_columns(self) -> None:
        from sema.engine.stage_utils import build_stage_c_prompt

        prompt = build_stage_c_prompt(
            columns_with_values=[
                {"column": "gender", "values": ["Male", "Female"]},
                {"column": "os_status", "values": ["0:LIVING"]},
            ],
            stage_a=_stage_a(),
            domain_context=None,
        )
        assert "gender" in prompt
        assert "os_status" in prompt


class TestStageCExecution:
    """Task 9.4: SemanticEngine.run_stage_c()."""

    def test_returns_c_results_for_flagged_columns(self) -> None:
        from sema.engine.semantic import SemanticEngine
        from sema.llm_client import LLMClient

        mock_llm_client = MagicMock(spec=LLMClient)
        mock_llm_client.invoke.return_value = StageCBatchResult(
            columns=[
                StageCResult(
                    column="gender",
                    decoded_categories=[
                        {"raw": "Male", "label": "male"},
                        {"raw": "Female", "label": "female"},
                    ],
                    uncertainty=0.1,
                ),
            ],
        )

        engine = SemanticEngine(
            llm_client=mock_llm_client, run_id="run-1",
        )
        table_meta = {
            "table_name": "patient",
            "table_ref": "unity://cat.sch.patient",
            "columns": [
                {"name": "gender", "data_type": "STRING",
                 "top_values": [{"value": "Male"}, {"value": "Female"}]},
            ],
        }
        stage_b = _stage_b([_b_col("gender", "categorical", needs_c=True)])

        results = engine.run_stage_c(table_meta, _stage_a(), stage_b)
        assert "gender" in results
        assert len(results["gender"].decoded_categories) == 2

    def test_skips_excluded_types(self) -> None:
        from sema.engine.semantic import SemanticEngine
        from sema.llm_client import LLMClient

        mock_llm_client = MagicMock(spec=LLMClient)

        engine = SemanticEngine(
            llm_client=mock_llm_client, run_id="run-1",
        )
        table_meta = {
            "table_name": "patient",
            "table_ref": "unity://cat.sch.patient",
            "columns": [
                {"name": "patient_id", "data_type": "STRING",
                 "top_values": [{"value": "P001"}]},
            ],
        }
        stage_b = _stage_b([
            _b_col("patient_id", "identifier", needs_c=True),
        ])

        results = engine.run_stage_c(table_meta, _stage_a(), stage_b)
        assert results == {}
        mock_llm_client.invoke.assert_not_called()

    def test_skips_unresolved_b_columns(self) -> None:
        from sema.engine.semantic import SemanticEngine
        from sema.llm_client import LLMClient

        mock_llm_client = MagicMock(spec=LLMClient)

        engine = SemanticEngine(
            llm_client=mock_llm_client, run_id="run-1",
        )
        table_meta = {
            "table_name": "patient",
            "table_ref": "unity://cat.sch.patient",
            "columns": [
                {"name": "gender", "data_type": "STRING",
                 "top_values": [{"value": "Male"}]},
            ],
        }
        # Stage B with unresolved gender column
        stage_b = StageBResult(
            status="B_PARTIAL",
            batch_results=[],
            raw_coverage=StageBCoverage(
                classified=0, total=1, pct=0.0,
            ),
            critical_coverage=StageBCoverage(
                classified=0, total=1, pct=0.0,
            ),
            unresolved_columns=[
                UnresolvedColumn(
                    column="gender",
                    reason="execution_failure",
                    tier="important",
                ),
            ],
        )

        results = engine.run_stage_c(table_meta, _stage_a(), stage_b)
        assert results == {}


class TestStageCMerge:
    """Task 9.5: C results feed into merge producing HAS_DECODED_VALUE."""

    def test_merge_with_c_results(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs

        stage_a = _stage_a()
        stage_b = _stage_b([_b_col("gender", "categorical")])
        c_results = {
            "gender": StageCResult(
                column="gender",
                decoded_categories=[
                    {"raw": "Male", "label": "male"},
                    {"raw": "Female", "label": "female"},
                ],
            ),
        }

        assertions = merge_stage_outputs(
            "unity://cat.sch.patient",
            stage_a, stage_b,
            c_results=c_results,
            run_id="run-1",
        )
        decoded = [
            a for a in assertions
            if a.predicate.value == "has_decoded_value"
        ]
        assert len(decoded) == 2

    def test_merge_without_c_no_decoded_values(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs

        assertions = merge_stage_outputs(
            "unity://cat.sch.patient",
            _stage_a(),
            _stage_b([_b_col("gender", "categorical")]),
            run_id="run-1",
        )
        decoded = [
            a for a in assertions
            if a.predicate.value == "has_decoded_value"
        ]
        assert len(decoded) == 0


class TestStageCPartialFailure:
    """Task 9.6: partial failure handling."""

    def test_partial_c_results_merged(self) -> None:
        from sema.engine.semantic import SemanticEngine
        from sema.llm_client import LLMClient, LLMStageError

        mock_llm_client = MagicMock(spec=LLMClient)

        # Batch call fails → falls back to per-column
        # Per-column: gender succeeds, os_status fails
        _stage_err = LLMStageError(
            table_ref="t",
            stage_name="L2 stage_c",
            step_errors=[("plain_invoke", ValueError("timeout"))],
        )
        mock_llm_client.invoke.side_effect = [
            _stage_err,  # batch call fails
            StageCResult(  # per-column: gender succeeds
                column="gender",
                decoded_categories=[
                    {"raw": "Male", "label": "male"},
                ],
            ),
            _stage_err,  # per-column: os_status fails
        ]

        engine = SemanticEngine(
            llm_client=mock_llm_client, run_id="run-1",
        )
        table_meta = {
            "table_name": "patient",
            "table_ref": "unity://cat.sch.patient",
            "columns": [
                {"name": "gender", "data_type": "STRING",
                 "top_values": [{"value": "Male"}]},
                {"name": "os_status", "data_type": "STRING",
                 "top_values": [{"value": "0:LIVING"}]},
            ],
        }
        stage_b = _stage_b([
            _b_col("gender", "categorical", needs_c=True),
            _b_col("os_status", "categorical", needs_c=True),
        ])

        results = engine.run_stage_c(table_meta, _stage_a(), stage_b)
        # gender succeeded, os_status failed
        assert "gender" in results
        assert "os_status" not in results
