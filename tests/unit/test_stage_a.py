"""Tests for Stage A: entity and grain hypothesis.

Covers: StageAResult schema validation, prompt construction,
response parsing via SemanticEngine.run_stage_a().
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.models.stages import StageAResult

pytestmark = pytest.mark.unit


# -- Fixtures ---------------------------------------------------------------

SAMPLE_TABLE: dict[str, Any] = {
    "table_ref": "unity://catalog.schema.data_mutations",
    "table_name": "data_mutations",
    "comment": "Somatic mutation calls per sample",
    "columns": [
        {"name": "patient_id", "data_type": "STRING", "comment": "Patient identifier"},
        {"name": "sample_id", "data_type": "STRING", "comment": None},
        {"name": "Hugo_Symbol", "data_type": "STRING", "comment": "HUGO gene symbol",
         "top_values": [
             {"value": "TP53"}, {"value": "KRAS"}, {"value": "EGFR"},
         ]},
        {"name": "Variant_Classification", "data_type": "STRING", "comment": None,
         "top_values": [
             {"value": "Missense_Mutation"}, {"value": "Silent"},
         ]},
        {"name": "t_alt_count", "data_type": "INT", "comment": "Tumor alt allele count"},
    ],
    "sample_rows": [
        {"patient_id": "P001", "sample_id": "S001", "Hugo_Symbol": "TP53",
         "Variant_Classification": "Missense_Mutation", "t_alt_count": 42},
    ],
}

NARROW_TABLE: dict[str, Any] = {
    "table_ref": "unity://catalog.schema.patient",
    "table_name": "patient",
    "comment": None,
    "columns": [
        {"name": "patient_id", "data_type": "STRING"},
        {"name": "age", "data_type": "INT"},
    ],
}


STAGE_A_RESPONSE: dict[str, Any] = {
    "primary_entity": "Somatic Mutation",
    "grain_hypothesis": "one row per variant call per sample",
    "synonyms": ["mutation call", "variant"],
    "secondary_entity_hints": ["gene", "protein change"],
    "ambiguity_flags": [],
    "confidence": 0.88,
}


# -- 2.1 StageAResult schema -----------------------------------------------

class TestStageAResultSchema:
    def test_valid_full_response(self) -> None:
        result = StageAResult(**STAGE_A_RESPONSE)
        assert result.primary_entity == "Somatic Mutation"
        assert result.grain_hypothesis == "one row per variant call per sample"
        assert result.secondary_entity_hints == ["gene", "protein change"]
        assert result.ambiguity_flags == []
        assert result.confidence == 0.88

    def test_minimal_required_fields(self) -> None:
        result = StageAResult(
            primary_entity="Patient",
            grain_hypothesis="one row per patient",
            confidence=0.9,
        )
        assert result.primary_entity == "Patient"
        assert result.secondary_entity_hints == []
        assert result.ambiguity_flags == []

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValueError):
            StageAResult(
                primary_entity="X", grain_hypothesis="y", confidence=1.5,
            )
        with pytest.raises(ValueError):
            StageAResult(
                primary_entity="X", grain_hypothesis="y", confidence=-0.1,
            )

    def test_ambiguity_flags_populated(self) -> None:
        result = StageAResult(
            primary_entity="Unknown",
            grain_hypothesis="unclear",
            ambiguity_flags=["mixed patient/sample granularity"],
            confidence=0.4,
        )
        assert len(result.ambiguity_flags) == 1

    def test_serialization_roundtrip(self) -> None:
        original = StageAResult(**STAGE_A_RESPONSE)
        data = original.model_dump()
        restored = StageAResult(**data)
        assert restored == original

    def test_synonyms_field(self) -> None:
        result = StageAResult(**STAGE_A_RESPONSE)
        assert result.synonyms == ["mutation call", "variant"]

    def test_synonyms_default_empty(self) -> None:
        result = StageAResult(
            primary_entity="X", grain_hypothesis="y", confidence=0.5,
        )
        assert result.synonyms == []

    def test_missing_primary_entity_raises(self) -> None:
        with pytest.raises(ValueError):
            StageAResult(grain_hypothesis="x", confidence=0.5)  # type: ignore[call-arg]


# -- 2.2 Stage A prompt construction ---------------------------------------

class TestStageAPrompt:
    def test_prompt_includes_table_name(self) -> None:
        from sema.engine.stage_utils import build_stage_a_prompt
        prompt = build_stage_a_prompt(SAMPLE_TABLE)
        assert "data_mutations" in prompt

    def test_prompt_includes_comment(self) -> None:
        from sema.engine.stage_utils import build_stage_a_prompt
        prompt = build_stage_a_prompt(SAMPLE_TABLE)
        assert "Somatic mutation calls per sample" in prompt

    def test_prompt_includes_column_names_and_types(self) -> None:
        from sema.engine.stage_utils import build_stage_a_prompt
        prompt = build_stage_a_prompt(SAMPLE_TABLE)
        assert "patient_id" in prompt
        assert "STRING" in prompt
        assert "Hugo_Symbol" in prompt

    def test_prompt_includes_sample_rows(self) -> None:
        from sema.engine.stage_utils import build_stage_a_prompt
        prompt = build_stage_a_prompt(SAMPLE_TABLE)
        assert "P001" in prompt

    def test_prompt_requests_stage_a_fields(self) -> None:
        from sema.engine.stage_utils import build_stage_a_prompt
        prompt = build_stage_a_prompt(SAMPLE_TABLE)
        assert "primary_entity" in prompt
        assert "grain_hypothesis" in prompt
        assert "synonyms" in prompt
        assert "secondary_entity_hints" in prompt
        assert "ambiguity_flags" in prompt
        assert "confidence" in prompt

    def test_prompt_does_not_request_properties(self) -> None:
        from sema.engine.stage_utils import build_stage_a_prompt
        prompt = build_stage_a_prompt(SAMPLE_TABLE)
        assert "canonical_property_label" not in prompt
        assert "semantic_type" not in prompt.split("secondary_entity_hints")[0]

    def test_prompt_no_comment_when_absent(self) -> None:
        from sema.engine.stage_utils import build_stage_a_prompt
        prompt = build_stage_a_prompt(NARROW_TABLE)
        assert "Comment:" not in prompt

    def test_prompt_domain_slot_empty_by_default(self) -> None:
        from sema.engine.stage_utils import build_stage_a_prompt
        prompt = build_stage_a_prompt(SAMPLE_TABLE)
        # No domain bias header when domain_context is None
        assert "domain" not in prompt.lower().split("columns")[0]

    def test_prompt_domain_slot_with_context(self) -> None:
        from sema.engine.stage_utils import build_stage_a_prompt
        from sema.models.domain import DomainContext
        ctx = DomainContext(declared_domain="healthcare", domain_source="user")
        prompt = build_stage_a_prompt(SAMPLE_TABLE, domain_context=ctx)
        # Domain slot is present but empty for now (step 3 activates it)
        # Just verify the function accepts the parameter without error
        assert "data_mutations" in prompt


# -- 2.3 SemanticEngine.run_stage_a() -------------------------------------

class TestRunStageA:
    def _make_engine(
        self, response: dict[str, Any] | None = None,
    ) -> Any:
        from sema.engine.semantic import SemanticEngine

        mock_client = MagicMock()
        if response is not None:
            mock_client.invoke.return_value = StageAResult(**response)
        return SemanticEngine(
            llm_client=mock_client, run_id="test-run",
        )

    def test_returns_stage_a_result(self) -> None:
        engine = self._make_engine(STAGE_A_RESPONSE)
        result = engine.run_stage_a(SAMPLE_TABLE)
        assert isinstance(result, StageAResult)
        assert result.primary_entity == "Somatic Mutation"
        assert result.grain_hypothesis == "one row per variant call per sample"

    def test_does_not_produce_assertions(self) -> None:
        engine = self._make_engine(STAGE_A_RESPONSE)
        result = engine.run_stage_a(SAMPLE_TABLE)
        # run_stage_a returns StageAResult, not assertions
        assert isinstance(result, StageAResult)
        assert not isinstance(result, list)

    def test_passes_correct_schema_to_llm_client(self) -> None:
        engine = self._make_engine(STAGE_A_RESPONSE)
        engine.run_stage_a(SAMPLE_TABLE)
        call_args = engine._llm_client.invoke.call_args
        assert call_args[0][1] is StageAResult

    def test_passes_table_ref_to_llm_client(self) -> None:
        engine = self._make_engine(STAGE_A_RESPONSE)
        engine.run_stage_a(SAMPLE_TABLE)
        call_kwargs = engine._llm_client.invoke.call_args[1]
        assert call_kwargs["table_ref"] == SAMPLE_TABLE["table_ref"]

    def test_passes_stage_name(self) -> None:
        engine = self._make_engine(STAGE_A_RESPONSE)
        engine.run_stage_a(SAMPLE_TABLE)
        call_kwargs = engine._llm_client.invoke.call_args[1]
        assert "stage_a" in call_kwargs["stage_name"].lower()

    def test_works_without_sample_rows(self) -> None:
        engine = self._make_engine(STAGE_A_RESPONSE)
        table = {**NARROW_TABLE}
        result = engine.run_stage_a(table)
        assert isinstance(result, StageAResult)

    def test_llm_stage_error_propagates(self) -> None:
        from sema.llm_client import LLMStageError
        from sema.engine.semantic import SemanticEngine

        mock_client = MagicMock()
        mock_client.invoke.side_effect = LLMStageError(
            table_ref="unity://test", stage_name="stage_a",
            step_errors=[("structured_output", ValueError("bad"))],
        )
        engine = SemanticEngine(llm_client=mock_client, run_id="test-run")
        with pytest.raises(LLMStageError):
            engine.run_stage_a(SAMPLE_TABLE)

    def test_accepts_domain_context(self) -> None:
        from sema.engine.semantic import SemanticEngine
        from sema.models.domain import DomainContext

        mock_client = MagicMock()
        mock_client.invoke.return_value = StageAResult(**STAGE_A_RESPONSE)
        ctx = DomainContext(declared_domain="healthcare", domain_source="user")
        engine = SemanticEngine(
            llm_client=mock_client, run_id="test-run",
            domain_context=ctx,
        )
        result = engine.run_stage_a(SAMPLE_TABLE)
        assert isinstance(result, StageAResult)
