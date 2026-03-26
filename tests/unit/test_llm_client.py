import json
import time
from unittest.mock import MagicMock, patch

import pytest

from pydantic import BaseModel, ConfigDict

pytestmark = pytest.mark.unit

from sema.llm_client import (
    LLMClient,
    LLMStageError,
    TableSummary,
    VocabularyDetection,
    SynonymExpansion,
    parse_llm_response,
    _is_transient_error,
)
from sema.engine.semantic import TableInterpretation


# ---------------------------------------------------------------------------
# Universal fallback parser tests (Task 2.1)
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_clean_json(self):
        raw = '{"entity_name": "Patient", "entity_description": "A patient record"}'
        result = parse_llm_response(raw, TableInterpretation)
        assert result.entity_name == "Patient"

    def test_markdown_fenced_json(self):
        raw = '```json\n{"entity_name": "Patient"}\n```'
        result = parse_llm_response(raw, TableInterpretation)
        assert result.entity_name == "Patient"

    def test_markdown_fenced_no_language(self):
        raw = '```\n{"entity_name": "Patient"}\n```'
        result = parse_llm_response(raw, TableInterpretation)
        assert result.entity_name == "Patient"

    def test_json_embedded_in_prose(self):
        raw = 'Here is the result:\n{"entity_name": "Patient", "entity_description": "desc"}\nLet me know if you need more.'
        result = parse_llm_response(raw, TableInterpretation)
        assert result.entity_name == "Patient"

    def test_key_normalization_to_lowercase(self):
        raw = '{"Entity_Name": "Patient", "Entity_Description": "desc"}'
        result = parse_llm_response(raw, TableInterpretation)
        assert result.entity_name == "Patient"

    def test_wrapper_key_unwrapping_result(self):
        raw = '{"result": {"entity_name": "Patient"}}'
        result = parse_llm_response(raw, TableInterpretation)
        assert result.entity_name == "Patient"

    def test_wrapper_key_unwrapping_data(self):
        raw = '{"data": {"entity_name": "Patient"}}'
        result = parse_llm_response(raw, TableInterpretation)
        assert result.entity_name == "Patient"

    def test_wrapper_key_unwrapping_response(self):
        raw = '{"response": {"entity_name": "Patient"}}'
        result = parse_llm_response(raw, TableInterpretation)
        assert result.entity_name == "Patient"

    def test_no_json_raises_error(self):
        raw = "I cannot help with that request."
        with pytest.raises(ValueError, match="No JSON found"):
            parse_llm_response(raw, TableInterpretation)

    def test_vocabulary_detection_schema(self):
        raw = '{"vocabulary": "ICD-10", "confidence": 0.95}'
        result = parse_llm_response(raw, VocabularyDetection)
        assert result.vocabulary == "ICD-10"
        assert result.confidence == 0.95

    def test_synonym_expansion_schema(self):
        raw = '{"synonyms": [{"term": "Cancer", "synonyms": ["Neoplasm", "Tumor"]}]}'
        result = parse_llm_response(raw, SynonymExpansion)
        assert len(result.synonyms) == 1
        assert result.synonyms[0]["term"] == "Cancer"


# ---------------------------------------------------------------------------
# Pydantic schema key normalization tests (Task 2.4)
# ---------------------------------------------------------------------------

class TestSchemaKeyNormalization:
    def test_table_interpretation_mixed_casing(self):
        raw = '{"Entity_Name": "Patient", "Properties": [{"Column": "id", "Name": "ID", "Semantic_Type": "identifier"}]}'
        result = parse_llm_response(raw, TableInterpretation)
        assert result.entity_name == "Patient"
        assert len(result.properties) == 1
        assert result.properties[0].column == "id"

    def test_vocabulary_detection_mixed_casing(self):
        raw = '{"Vocabulary": "AJCC Staging", "Confidence": 0.85}'
        result = parse_llm_response(raw, VocabularyDetection)
        assert result.vocabulary == "AJCC Staging"

    def test_table_summary_mixed_casing(self):
        raw = '{"Entity_Name": "Diagnosis", "Entity_Description": "desc", "Synonyms": ["dx"]}'
        result = parse_llm_response(raw, TableSummary)
        assert result.entity_name == "Diagnosis"
        assert result.synonyms == ["dx"]


# ---------------------------------------------------------------------------
# LLMClient fallback chain tests (Task 2.2)
# ---------------------------------------------------------------------------

class TestLLMClientFallbackChain:
    def _make_client(self, llm, **kwargs):
        return LLMClient(llm, retry_max_attempts=1, **kwargs)

    def test_structured_output_success(self):
        llm = MagicMock()
        structured_llm = MagicMock()
        expected = TableInterpretation(entity_name="Patient")
        structured_llm.invoke.return_value = expected
        llm.with_structured_output.return_value = structured_llm

        client = self._make_client(llm, use_structured_output="true")
        result = client.invoke("test prompt", TableInterpretation)
        assert result.entity_name == "Patient"

    def test_structured_output_fail_fallback_parser_success(self):
        llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.invoke.side_effect = Exception("structured output failed")
        llm.with_structured_output.return_value = structured_llm

        response = MagicMock()
        response.content = '{"entity_name": "Patient"}'
        llm.invoke.return_value = response

        client = self._make_client(llm, use_structured_output="true")
        result = client.invoke("test prompt", TableInterpretation)
        assert result.entity_name == "Patient"

    def test_all_steps_fail_raises_llm_stage_error(self):
        llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.invoke.side_effect = Exception("structured failed")
        llm.with_structured_output.return_value = structured_llm

        response = MagicMock()
        response.content = "not valid json at all"
        llm.invoke.return_value = response

        client = self._make_client(llm, use_structured_output="true")
        with pytest.raises(LLMStageError) as exc_info:
            client.invoke(
                "test prompt",
                TableInterpretation,
                table_ref="unity://cdm.clinical.tbl",
                stage_name="L2 semantic",
            )
        assert exc_info.value.table_ref == "unity://cdm.clinical.tbl"
        assert exc_info.value.stage_name == "L2 semantic"
        assert len(exc_info.value.step_errors) == 2  # structured + plain

    def test_all_steps_fail_with_simplified_prompt(self):
        llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.invoke.side_effect = Exception("fail")
        llm.with_structured_output.return_value = structured_llm

        response = MagicMock()
        response.content = "garbage"
        llm.invoke.return_value = response

        client = self._make_client(llm, use_structured_output="true")
        with pytest.raises(LLMStageError) as exc_info:
            client.invoke(
                "test prompt",
                TableInterpretation,
                simplified_prompt="simple prompt",
                table_ref="ref",
                stage_name="test",
            )
        assert len(exc_info.value.step_errors) == 3  # structured + plain + simplified

    def test_simplified_prompt_succeeds_after_earlier_failures(self):
        llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.invoke.side_effect = Exception("fail")
        llm.with_structured_output.return_value = structured_llm

        call_count = [0]
        def invoke_side_effect(prompt):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                response.content = "garbage"
            else:
                response.content = '{"entity_name": "Patient"}'
            return response

        llm.invoke.side_effect = invoke_side_effect

        client = self._make_client(llm, use_structured_output="true")
        result = client.invoke(
            "complex prompt",
            TableInterpretation,
            simplified_prompt="simple prompt",
        )
        assert result.entity_name == "Patient"


# ---------------------------------------------------------------------------
# Provider detection tests (Task 2.3)
# ---------------------------------------------------------------------------

class TestProviderDetection:
    def test_detects_structured_output_support(self):
        llm = MagicMock()
        llm.with_structured_output = MagicMock()
        client = LLMClient(llm, retry_max_attempts=1)
        assert client._supports_structured is True

    def test_detects_no_structured_output_support(self):
        llm = MagicMock(spec=[])  # no with_structured_output
        client = LLMClient(llm, retry_max_attempts=1)
        assert client._supports_structured is False

    def test_no_structured_output_skips_step_1(self):
        llm = MagicMock(spec=["invoke"])
        response = MagicMock()
        response.content = '{"entity_name": "Patient"}'
        llm.invoke.return_value = response

        client = LLMClient(llm, retry_max_attempts=1)
        result = client.invoke("prompt", TableInterpretation)
        assert result.entity_name == "Patient"
        # with_structured_output was never called
        assert not hasattr(llm, "with_structured_output") or not llm.with_structured_output.called


# ---------------------------------------------------------------------------
# Retry logic tests (Task 3.1)
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def test_rate_limit_triggers_retry(self):
        llm = MagicMock(spec=["invoke"])
        rate_limit_error = Exception("rate limit")
        rate_limit_error.status_code = 429

        call_count = [0]
        def invoke_side_effect(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                raise rate_limit_error
            response = MagicMock()
            response.content = '{"entity_name": "Patient"}'
            return response

        llm.invoke.side_effect = invoke_side_effect

        client = LLMClient(llm, retry_max_attempts=3, retry_base_delay=0.01)
        result = client.invoke("prompt", TableInterpretation)
        assert result.entity_name == "Patient"
        assert call_count[0] == 2

    def test_auth_error_not_retried(self):
        llm = MagicMock(spec=["invoke"])
        auth_error = Exception("unauthorized")
        auth_error.status_code = 401
        llm.invoke.side_effect = auth_error

        client = LLMClient(llm, retry_max_attempts=3, retry_base_delay=0.01)
        with pytest.raises(LLMStageError):
            client.invoke("prompt", TableInterpretation, table_ref="ref", stage_name="test")
        # Only called once (no retry)
        assert llm.invoke.call_count == 1

    def test_forbidden_error_not_retried(self):
        llm = MagicMock(spec=["invoke"])
        forbidden_error = Exception("forbidden")
        forbidden_error.status_code = 403
        llm.invoke.side_effect = forbidden_error

        client = LLMClient(llm, retry_max_attempts=3, retry_base_delay=0.01)
        with pytest.raises(LLMStageError):
            client.invoke("prompt", TableInterpretation, table_ref="ref", stage_name="test")
        assert llm.invoke.call_count == 1

    def test_parse_failure_falls_to_simplified_prompt(self):
        """Parse failures in step 2 cause progression to step 3 (simplified prompt)."""
        llm = MagicMock(spec=["invoke"])
        call_count = [0]
        def invoke_side_effect(prompt):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                response.content = "not json at all"  # step 2 fails parse
            else:
                response.content = '{"entity_name": "Patient"}'  # step 3 succeeds
            return response

        llm.invoke.side_effect = invoke_side_effect

        client = LLMClient(llm, retry_max_attempts=1, retry_base_delay=0.01)
        result = client.invoke(
            "complex prompt",
            TableInterpretation,
            simplified_prompt="simple prompt",
        )
        assert result.entity_name == "Patient"
        assert call_count[0] == 2

    def test_max_attempts_exhausted(self):
        llm = MagicMock(spec=["invoke"])
        rate_error = Exception("rate limit")
        rate_error.status_code = 429
        llm.invoke.side_effect = rate_error

        client = LLMClient(llm, retry_max_attempts=3, retry_base_delay=0.01)
        with pytest.raises(LLMStageError):
            client.invoke("prompt", TableInterpretation, table_ref="ref", stage_name="test")
        assert llm.invoke.call_count == 3  # 3 attempts in step 2 (no structured output)


# ---------------------------------------------------------------------------
# Backoff timing tests (Task 3.2)
# ---------------------------------------------------------------------------

class TestBackoffTiming:
    def test_exponential_delay_pattern(self):
        llm = MagicMock(spec=["invoke"])
        rate_error = Exception("rate limit")
        rate_error.status_code = 429
        llm.invoke.side_effect = rate_error

        sleep_times = []
        original_sleep = time.sleep
        with patch("sema.llm_client.time.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda t: sleep_times.append(t)
            client = LLMClient(
                llm, retry_max_attempts=3, retry_base_delay=2.0,
                retry_multiplier=2.0, retry_jitter=0.0,
            )
            with pytest.raises(LLMStageError):
                client.invoke("prompt", TableInterpretation, table_ref="ref", stage_name="test")

        # 2 sleeps (between attempt 1→2 and 2→3)
        assert len(sleep_times) == 2
        assert sleep_times[0] == pytest.approx(2.0, abs=0.1)  # base * 2^0
        assert sleep_times[1] == pytest.approx(4.0, abs=0.1)  # base * 2^1

    def test_jitter_within_range(self):
        llm = MagicMock(spec=["invoke"])
        rate_error = Exception("rate limit")
        rate_error.status_code = 429
        llm.invoke.side_effect = rate_error

        sleep_times = []
        with patch("sema.llm_client.time.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda t: sleep_times.append(t)
            client = LLMClient(
                llm, retry_max_attempts=3, retry_base_delay=2.0,
                retry_multiplier=2.0, retry_jitter=0.5,
            )
            with pytest.raises(LLMStageError):
                client.invoke("prompt", TableInterpretation, table_ref="ref", stage_name="test")

        assert len(sleep_times) == 2
        # First delay: 2.0 ± 0.5 → [1.5, 2.5]
        assert 1.5 <= sleep_times[0] <= 2.5
        # Second delay: 4.0 ± 0.5 → [3.5, 4.5]
        assert 3.5 <= sleep_times[1] <= 4.5


# ---------------------------------------------------------------------------
# Configurable retry_max_attempts tests (Task 3.3)
# ---------------------------------------------------------------------------

class TestConfigurableRetry:
    def test_custom_retry_count(self):
        llm = MagicMock(spec=["invoke"])
        rate_error = Exception("rate limit")
        rate_error.status_code = 429
        llm.invoke.side_effect = rate_error

        client = LLMClient(llm, retry_max_attempts=5, retry_base_delay=0.01)
        with pytest.raises(LLMStageError):
            client.invoke("prompt", TableInterpretation, table_ref="ref", stage_name="test")
        assert llm.invoke.call_count == 5

    def test_retries_disabled(self):
        llm = MagicMock(spec=["invoke"])
        rate_error = Exception("rate limit")
        rate_error.status_code = 429
        llm.invoke.side_effect = rate_error

        client = LLMClient(llm, retry_max_attempts=1, retry_base_delay=0.01)
        with pytest.raises(LLMStageError):
            client.invoke("prompt", TableInterpretation, table_ref="ref", stage_name="test")
        assert llm.invoke.call_count == 1


# ---------------------------------------------------------------------------
# LLMStageError tests (Task 2.5)
# ---------------------------------------------------------------------------

class TestLLMStageError:
    def test_error_contains_context(self):
        error = LLMStageError(
            table_ref="unity://cdm.clinical.tbl",
            stage_name="L2 semantic",
            step_errors=[
                ("structured_output", ValueError("bad json")),
                ("plain_invoke", ValueError("no json found")),
            ],
        )
        assert error.table_ref == "unity://cdm.clinical.tbl"
        assert error.stage_name == "L2 semantic"
        assert len(error.step_errors) == 2
        assert "L2 semantic" in str(error)
        assert "structured_output" in str(error)

    def test_transient_error_detection(self):
        rate_err = Exception("rate limit")
        rate_err.status_code = 429
        assert _is_transient_error(rate_err) is True

        auth_err = Exception("unauthorized")
        auth_err.status_code = 401
        assert _is_transient_error(auth_err) is False

        parse_err = json.JSONDecodeError("bad", "", 0)
        assert _is_transient_error(parse_err) is True

        generic_err = Exception("something else")
        assert _is_transient_error(generic_err) is False
