"""Tests for SemanticEngine.interpret_table (thin wrapper over staged pipeline).

Deep staged-pipeline coverage lives in test_stage_a.py, test_stage_b.py,
test_stage_c.py, and test_merge_stages.py — this module only verifies
that interpret_table() delegates and that LLMStageError propagates.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sema.engine.semantic import SemanticEngine
from sema.llm_client import LLMClient, LLMStageError
from sema.models.assertions import AssertionPredicate
from sema.models.stages import (
    StageAResult,
    StageBBatchResult,
    StageBColumnResult,
)

pytestmark = pytest.mark.unit


def _sample_meta() -> dict:
    return {
        "table_ref": "unity://cdm.clinical.patient",
        "table_name": "patient",
        "columns": [{"name": "patient_id", "data_type": "STRING"}],
        "sample_rows": [],
        "comment": None,
    }


class TestInterpretTable:
    def test_delegates_to_staged_pipeline(self) -> None:
        client = MagicMock(spec=LLMClient)
        client.invoke.side_effect = [
            StageAResult(
                primary_entity="Patient",
                grain_hypothesis="one row per patient",
                confidence=0.9,
            ),
            StageBBatchResult(columns=[
                StageBColumnResult(
                    column="patient_id",
                    canonical_property_label="patient identifier",
                    semantic_type="identifier",
                    entity_role="primary_key",
                    needs_stage_c=False,
                ),
            ]),
        ]
        engine = SemanticEngine(llm_client=client, run_id="test-run")

        assertions = engine.interpret_table(_sample_meta())

        entity = [a for a in assertions
                  if a.predicate == AssertionPredicate.HAS_ENTITY_NAME]
        assert len(entity) == 1
        assert entity[0].payload["value"] == "Patient"

        props = [a for a in assertions
                 if a.predicate == AssertionPredicate.HAS_PROPERTY_NAME]
        assert len(props) == 1
        assert props[0].payload["value"] == "patient identifier"

    def test_llm_stage_error_propagates(self) -> None:
        client = MagicMock(spec=LLMClient)
        client.invoke.side_effect = LLMStageError(
            table_ref="unity://cdm.clinical.patient",
            stage_name="L2 stage_a",
            step_errors=[("structured_output", ValueError("fail"))],
        )
        engine = SemanticEngine(llm_client=client, run_id="test-run")

        with pytest.raises(LLMStageError) as exc_info:
            engine.interpret_table(_sample_meta())
        assert exc_info.value.stage_name == "L2 stage_a"

    def test_no_vocabulary_match_emitted_from_l2(self) -> None:
        """Per design §2a — L3 owns vocabulary_match assertions."""
        client = MagicMock(spec=LLMClient)
        client.invoke.side_effect = [
            StageAResult(
                primary_entity="Patient",
                grain_hypothesis="one row per patient",
                confidence=0.9,
            ),
            StageBBatchResult(columns=[
                StageBColumnResult(
                    column="patient_id",
                    canonical_property_label="patient identifier",
                    semantic_type="identifier",
                    entity_role="primary_key",
                    needs_stage_c=False,
                    candidate_vocab_families=["identifier namespace"],
                ),
            ]),
        ]
        engine = SemanticEngine(llm_client=client, run_id="test-run")

        assertions = engine.interpret_table(_sample_meta())
        vocab = [a for a in assertions
                 if a.predicate == AssertionPredicate.VOCABULARY_MATCH]
        assert vocab == []
