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


# ---------------------------------------------------------------------------
# Universal fallback parser tests (Task 2.1)
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_clean_json(self):
        raw = '{"entity_name": "Patient", "entity_description": "A patient record"}'
        result = parse_llm_response(raw, TableSummary)
        assert result.entity_name == "Patient"

    def test_markdown_fenced_json(self):
        raw = '```json\n{"entity_name": "Patient"}\n```'
        result = parse_llm_response(raw, TableSummary)
        assert result.entity_name == "Patient"

    def test_markdown_fenced_no_language(self):
        raw = '```\n{"entity_name": "Patient"}\n```'
        result = parse_llm_response(raw, TableSummary)
        assert result.entity_name == "Patient"

    def test_json_embedded_in_prose(self):
        raw = 'Here is the result:\n{"entity_name": "Patient", "entity_description": "desc"}\nLet me know if you need more.'
        result = parse_llm_response(raw, TableSummary)
        assert result.entity_name == "Patient"

    def test_key_normalization_to_lowercase(self):
        raw = '{"Entity_Name": "Patient", "Entity_Description": "desc"}'
        result = parse_llm_response(raw, TableSummary)
        assert result.entity_name == "Patient"

    def test_wrapper_key_unwrapping_result(self):
        raw = '{"result": {"entity_name": "Patient"}}'
        result = parse_llm_response(raw, TableSummary)
        assert result.entity_name == "Patient"

    def test_wrapper_key_unwrapping_data(self):
        raw = '{"data": {"entity_name": "Patient"}}'
        result = parse_llm_response(raw, TableSummary)
        assert result.entity_name == "Patient"

    def test_wrapper_key_unwrapping_response(self):
        raw = '{"response": {"entity_name": "Patient"}}'
        result = parse_llm_response(raw, TableSummary)
        assert result.entity_name == "Patient"

    def test_no_json_raises_error(self):
        raw = "I cannot help with that request."
        with pytest.raises(ValueError, match="No JSON found"):
            parse_llm_response(raw, TableSummary)

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
    def test_table_summary_mixed_casing(self):
        raw = '{"Entity_Name": "Patient", "Entity_Description": "A patient"}'
        result = parse_llm_response(raw, TableSummary)
        assert result.entity_name == "Patient"
        assert result.entity_description == "A patient"

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
        expected = TableSummary(entity_name="Patient")
        structured_llm.invoke.return_value = expected
        llm.with_structured_output.return_value = structured_llm

        client = self._make_client(llm, use_structured_output="true")
        result = client.invoke("test prompt", TableSummary)
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
        result = client.invoke("test prompt", TableSummary)
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
                TableSummary,
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
                TableSummary,
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
            TableSummary,
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
        result = client.invoke("prompt", TableSummary)
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

        client = LLMClient(
            llm, retry_max_attempts=3, retry_base_delay=0.01,
            rate_limit_base_delay=0.01,
        )
        result = client.invoke("prompt", TableSummary)
        assert result.entity_name == "Patient"
        assert call_count[0] == 2

    def test_auth_error_not_retried(self):
        llm = MagicMock(spec=["invoke"])
        auth_error = Exception("unauthorized")
        auth_error.status_code = 401
        llm.invoke.side_effect = auth_error

        client = LLMClient(llm, retry_max_attempts=3, retry_base_delay=0.01)
        with pytest.raises(LLMStageError):
            client.invoke("prompt", TableSummary, table_ref="ref", stage_name="test")
        # Only called once (no retry)
        assert llm.invoke.call_count == 1

    def test_forbidden_error_not_retried(self):
        llm = MagicMock(spec=["invoke"])
        forbidden_error = Exception("forbidden")
        forbidden_error.status_code = 403
        llm.invoke.side_effect = forbidden_error

        client = LLMClient(llm, retry_max_attempts=3, retry_base_delay=0.01)
        with pytest.raises(LLMStageError):
            client.invoke("prompt", TableSummary, table_ref="ref", stage_name="test")
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
            TableSummary,
            simplified_prompt="simple prompt",
        )
        assert result.entity_name == "Patient"
        assert call_count[0] == 2

    def test_max_attempts_exhausted(self):
        llm = MagicMock(spec=["invoke"])
        rate_error = Exception("rate limit")
        rate_error.status_code = 429
        llm.invoke.side_effect = rate_error

        client = LLMClient(
            llm, retry_max_attempts=3, retry_base_delay=0.01,
            rate_limit_base_delay=0.01,
        )
        with pytest.raises(LLMStageError):
            client.invoke("prompt", TableSummary, table_ref="ref", stage_name="test")
        assert llm.invoke.call_count == 3  # 3 attempts in step 2 (no structured output)


# ---------------------------------------------------------------------------
# Backoff timing tests (Task 3.2)
# ---------------------------------------------------------------------------

class TestBackoffTiming:
    def test_exponential_delay_pattern_for_non_rate_limit_transient(self):
        llm = MagicMock(spec=["invoke"])
        server_error = Exception("internal server error")
        server_error.status_code = 500
        llm.invoke.side_effect = server_error

        sleep_times = []
        with patch("sema.llm_client.time.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda t: sleep_times.append(t)
            client = LLMClient(
                llm, retry_max_attempts=3, retry_base_delay=2.0,
                retry_multiplier=2.0, retry_jitter=0.0,
            )
            with pytest.raises(LLMStageError):
                client.invoke("prompt", TableSummary, table_ref="ref", stage_name="test")

        assert len(sleep_times) == 2
        assert sleep_times[0] == pytest.approx(2.0, abs=0.1)
        assert sleep_times[1] == pytest.approx(4.0, abs=0.1)

    def test_jitter_within_range_for_non_rate_limit_transient(self):
        llm = MagicMock(spec=["invoke"])
        server_error = Exception("internal server error")
        server_error.status_code = 500
        llm.invoke.side_effect = server_error

        sleep_times = []
        with patch("sema.llm_client.time.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda t: sleep_times.append(t)
            client = LLMClient(
                llm, retry_max_attempts=3, retry_base_delay=2.0,
                retry_multiplier=2.0, retry_jitter=0.5,
            )
            with pytest.raises(LLMStageError):
                client.invoke("prompt", TableSummary, table_ref="ref", stage_name="test")

        assert len(sleep_times) == 2
        assert 1.5 <= sleep_times[0] <= 2.5
        assert 3.5 <= sleep_times[1] <= 4.5

    def test_rate_limit_uses_long_backoff_schedule(self):
        llm = MagicMock(spec=["invoke"])
        rate_error = Exception("REQUEST_LIMIT_EXCEEDED: rate limit")
        rate_error.status_code = 429
        llm.invoke.side_effect = rate_error

        sleep_times = []
        with patch("sema.llm_client.time.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda t: sleep_times.append(t)
            client = LLMClient(
                llm, retry_max_attempts=3, retry_base_delay=2.0,
                retry_multiplier=2.0, retry_jitter=0.0,
            )
            with pytest.raises(LLMStageError):
                client.invoke("prompt", TableSummary, table_ref="ref", stage_name="test")

        assert len(sleep_times) == 2
        assert sleep_times[0] >= 10.0
        assert sleep_times[1] >= 30.0


@pytest.mark.unit
class TestRateLimitClassification:
    def test_request_limit_exceeded_message_classified_as_rate_limit(self):
        from sema.llm_client import _is_rate_limit_error

        err = Exception("REQUEST_LIMIT_EXCEEDED: Exceeded workspace tokens/min")
        assert _is_rate_limit_error(err) is True

    def test_429_status_code_classified_as_rate_limit(self):
        from sema.llm_client import _is_rate_limit_error

        err = Exception("nope")
        err.status_code = 429
        assert _is_rate_limit_error(err) is True

    def test_500_not_classified_as_rate_limit(self):
        from sema.llm_client import _is_rate_limit_error

        err = Exception("internal server error")
        err.status_code = 500
        assert _is_rate_limit_error(err) is False


@pytest.mark.unit
class TestCircuitBreakerInteraction:
    def test_pure_rate_limit_failure_does_not_record_circuit_breaker_failure(self):
        llm = MagicMock(spec=["invoke"])
        rate_error = Exception("REQUEST_LIMIT_EXCEEDED")
        rate_error.status_code = 429
        llm.invoke.side_effect = rate_error
        breaker = MagicMock()
        breaker.check.return_value = None

        with patch("sema.llm_client.time.sleep"):
            client = LLMClient(
                llm, retry_max_attempts=2, retry_base_delay=0.01,
                circuit_breaker=breaker,
            )
            with pytest.raises(LLMStageError):
                client.invoke("prompt", TableSummary, table_ref="ref", stage_name="test")

        breaker.record_failure.assert_not_called()

    def test_non_rate_limit_failure_still_records_circuit_breaker_failure(self):
        llm = MagicMock(spec=["invoke"])
        server_error = Exception("internal server error")
        server_error.status_code = 500
        llm.invoke.side_effect = server_error
        breaker = MagicMock()
        breaker.check.return_value = None

        with patch("sema.llm_client.time.sleep"):
            client = LLMClient(
                llm, retry_max_attempts=2, retry_base_delay=0.01,
                circuit_breaker=breaker,
            )
            with pytest.raises(LLMStageError):
                client.invoke("prompt", TableSummary, table_ref="ref", stage_name="test")

        breaker.record_failure.assert_called_once()

    def test_success_records_circuit_breaker_success(self):
        llm = MagicMock(spec=["invoke"])
        response = MagicMock()
        response.content = '{"entity_name": "Patient"}'
        llm.invoke.return_value = response
        breaker = MagicMock()
        breaker.check.return_value = None

        client = LLMClient(
            llm, retry_max_attempts=1, retry_base_delay=0.01,
            circuit_breaker=breaker,
        )
        result = client.invoke("prompt", TableSummary)

        assert result.entity_name == "Patient"
        breaker.record_success.assert_called_once()
        breaker.record_failure.assert_not_called()


# ---------------------------------------------------------------------------
# Configurable retry_max_attempts tests (Task 3.3)
# ---------------------------------------------------------------------------

class TestConfigurableRetry:
    def test_custom_retry_count(self):
        llm = MagicMock(spec=["invoke"])
        rate_error = Exception("rate limit")
        rate_error.status_code = 429
        llm.invoke.side_effect = rate_error

        client = LLMClient(
            llm, retry_max_attempts=5, retry_base_delay=0.01,
            rate_limit_base_delay=0.01,
        )
        with pytest.raises(LLMStageError):
            client.invoke("prompt", TableSummary, table_ref="ref", stage_name="test")
        assert llm.invoke.call_count == 5

    def test_retries_disabled(self):
        llm = MagicMock(spec=["invoke"])
        rate_error = Exception("rate limit")
        rate_error.status_code = 429
        llm.invoke.side_effect = rate_error

        client = LLMClient(
            llm, retry_max_attempts=1, retry_base_delay=0.01,
            rate_limit_base_delay=0.01,
        )
        with pytest.raises(LLMStageError):
            client.invoke("prompt", TableSummary, table_ref="ref", stage_name="test")
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


# ---------------------------------------------------------------------------
# Service-health classification tests (Section 1 — opt-in breaker semantics)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestServiceHealthClassification:
    def test_http_503_is_service_health(self):
        from sema.llm_client import _is_service_health_failure
        err = Exception("Service Unavailable")
        err.status_code = 503
        assert _is_service_health_failure(err) is True

    def test_http_500_is_service_health(self):
        from sema.llm_client import _is_service_health_failure
        err = Exception("Internal Server Error")
        err.status_code = 500
        assert _is_service_health_failure(err) is True

    def test_http_502_is_service_health(self):
        from sema.llm_client import _is_service_health_failure
        err = Exception("Bad Gateway")
        err.status_code = 502
        assert _is_service_health_failure(err) is True

    def test_http_504_is_service_health(self):
        from sema.llm_client import _is_service_health_failure
        err = Exception("Gateway Timeout")
        err.status_code = 504
        assert _is_service_health_failure(err) is True

    def test_http_429_is_not_service_health(self):
        from sema.llm_client import _is_service_health_failure
        err = Exception("Too Many Requests")
        err.status_code = 429
        assert _is_service_health_failure(err) is False

    def test_rate_limit_message_is_not_service_health(self):
        from sema.llm_client import _is_service_health_failure
        err = Exception("REQUEST_LIMIT_EXCEEDED: tokens/min")
        assert _is_service_health_failure(err) is False

    def test_json_decode_error_is_not_service_health(self):
        from sema.llm_client import _is_service_health_failure
        err = json.JSONDecodeError("bad json", "", 0)
        assert _is_service_health_failure(err) is False

    def test_pydantic_validation_error_is_not_service_health(self):
        from pydantic import ValidationError as PydValidationError
        from sema.llm_client import _is_service_health_failure

        class _S(BaseModel):
            x: int
        try:
            _S.model_validate({"x": "not-an-int"})
        except PydValidationError as exc:
            assert _is_service_health_failure(exc) is False
            return
        pytest.fail("ValidationError should have been raised")

    def test_value_error_no_json_found_is_not_service_health(self):
        from sema.llm_client import _is_service_health_failure
        err = ValueError("No JSON found in LLM response: ...")
        assert _is_service_health_failure(err) is False

    def test_value_error_could_not_parse_is_not_service_health(self):
        from sema.llm_client import _is_service_health_failure
        err = ValueError("Could not parse LLM response into TableSummary: ...")
        assert _is_service_health_failure(err) is False

    def test_value_error_b_failed_is_not_service_health_defensive(self):
        """Defensive: B_FAILED ValueError currently does not flow through
        LLMClient.invoke (raised at semantic.py:450, outside invoke).
        If a future refactor routes it through, the breaker must not trip."""
        from sema.llm_client import _is_service_health_failure
        err = ValueError("B_FAILED: raw=0.65")
        assert _is_service_health_failure(err) is False

    def test_unknown_exception_is_not_service_health(self):
        from sema.llm_client import _is_service_health_failure
        err = RuntimeError("something unexpected")
        assert _is_service_health_failure(err) is False

    def test_connection_error_is_service_health(self):
        from sema.llm_client import _is_service_health_failure
        err = ConnectionError("Connection refused")
        assert _is_service_health_failure(err) is True

    def test_timeout_error_is_service_health(self):
        from sema.llm_client import _is_service_health_failure
        err = TimeoutError("socket read timed out")
        assert _is_service_health_failure(err) is True


@pytest.mark.unit
class TestCircuitBreakerOptInService:
    """Section 1: breaker only records service-health failures.

    The cascade-pollution path is per-batch invokes inside
    `_invoke_stage_b_batch`: content failures from `LLMClient.invoke`."""

    def _client_for(self, llm, breaker):
        return LLMClient(
            llm, retry_max_attempts=1, retry_base_delay=0.01,
            circuit_breaker=breaker,
        )

    def test_503_step_error_records_breaker(self):
        llm = MagicMock(spec=["invoke"])
        err = Exception("Service Unavailable")
        err.status_code = 503
        llm.invoke.side_effect = err
        breaker = MagicMock()
        breaker.check.return_value = None

        with patch("sema.llm_client.time.sleep"):
            client = self._client_for(llm, breaker)
            with pytest.raises(LLMStageError):
                client.invoke(
                    "p", TableSummary, table_ref="r", stage_name="s",
                )
        breaker.record_failure.assert_called_once()

    def test_429_step_error_does_not_record(self):
        llm = MagicMock(spec=["invoke"])
        err = Exception("rate limit")
        err.status_code = 429
        llm.invoke.side_effect = err
        breaker = MagicMock()
        breaker.check.return_value = None

        with patch("sema.llm_client.time.sleep"):
            client = self._client_for(llm, breaker)
            with pytest.raises(LLMStageError):
                client.invoke(
                    "p", TableSummary, table_ref="r", stage_name="s",
                )
        breaker.record_failure.assert_not_called()

    def test_json_decode_step_error_does_not_record(self):
        """Per-batch content-failure storm must not trip the breaker."""
        llm = MagicMock(spec=["invoke"])
        response = MagicMock()
        response.content = "not valid json at all"
        llm.invoke.return_value = response
        breaker = MagicMock()
        breaker.check.return_value = None

        client = self._client_for(llm, breaker)
        with pytest.raises(LLMStageError):
            client.invoke(
                "p", TableSummary, table_ref="r", stage_name="s",
            )
        breaker.record_failure.assert_not_called()

    def test_value_error_no_json_does_not_record(self):
        llm = MagicMock(spec=["invoke"])
        response = MagicMock()
        response.content = "i cannot help with that request"
        llm.invoke.return_value = response
        breaker = MagicMock()
        breaker.check.return_value = None

        client = self._client_for(llm, breaker)
        with pytest.raises(LLMStageError):
            client.invoke(
                "p", TableSummary, table_ref="r", stage_name="s",
            )
        breaker.record_failure.assert_not_called()

    def test_mixed_503_and_json_decode_records(self):
        """When step_errors carry [(structured, JSONDecodeError),
        (plain, 503)], the breaker SHALL record because at least one is
        service-health."""
        llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.invoke.side_effect = json.JSONDecodeError(
            "bad", "", 0,
        )
        llm.with_structured_output.return_value = structured_llm

        err503 = Exception("Service Unavailable")
        err503.status_code = 503
        llm.invoke.side_effect = err503

        breaker = MagicMock()
        breaker.check.return_value = None

        with patch("sema.llm_client.time.sleep"):
            client = LLMClient(
                llm, retry_max_attempts=1, retry_base_delay=0.01,
                use_structured_output="true",
                circuit_breaker=breaker,
            )
            with pytest.raises(LLMStageError):
                client.invoke(
                    "p", TableSummary, table_ref="r", stage_name="s",
                )
        breaker.record_failure.assert_called_once()

    def test_unknown_exception_does_not_record(self):
        llm = MagicMock(spec=["invoke"])
        llm.invoke.side_effect = RuntimeError("mystery")
        breaker = MagicMock()
        breaker.check.return_value = None

        client = self._client_for(llm, breaker)
        with pytest.raises(LLMStageError):
            client.invoke(
                "p", TableSummary, table_ref="r", stage_name="s",
            )
        breaker.record_failure.assert_not_called()

    def test_b_failed_synthetic_value_error_does_not_record_defensive(self):
        """Defensive: synthetic `B_FAILED` `LLMStageError` currently never
        flows through `LLMClient.invoke` (raised at semantic.py:450,
        outside invoke). If a future refactor routes it through, the
        breaker SHALL still not trip on the contained ValueError."""
        from sema.llm_client import _is_service_health_failure

        synthetic = ValueError("B_FAILED: raw=0.65")
        assert _is_service_health_failure(synthetic) is False

        stage_err = LLMStageError(
            table_ref="r",
            stage_name="L2 stage_b",
            step_errors=[("stage_b", synthetic)],
        )
        assert all(
            not _is_service_health_failure(e)
            for _, e in stage_err.step_errors
        )


@pytest.mark.unit
class TestRetryPolicyUnchanged:
    """Section 1.5: keep retry-backoff selection independent of breaker
    classification — a service-health 503 still uses the retry backoff
    schedule, not the rate-limit one."""

    def test_503_uses_short_retry_backoff_not_rate_limit_schedule(self):
        llm = MagicMock(spec=["invoke"])
        err = Exception("Service Unavailable")
        err.status_code = 503
        llm.invoke.side_effect = err

        sleep_times: list[float] = []
        with patch("sema.llm_client.time.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda t: sleep_times.append(t)
            client = LLMClient(
                llm, retry_max_attempts=3, retry_base_delay=2.0,
                retry_multiplier=2.0, retry_jitter=0.0,
                rate_limit_base_delay=10.0,
            )
            with pytest.raises(LLMStageError):
                client.invoke(
                    "p", TableSummary, table_ref="r", stage_name="s",
                )

        assert len(sleep_times) == 2
        # Short retry schedule, not the long rate-limit one
        assert sleep_times[0] == pytest.approx(2.0, abs=0.1)
        assert sleep_times[1] == pytest.approx(4.0, abs=0.1)

    def test_429_still_uses_long_rate_limit_backoff(self):
        llm = MagicMock(spec=["invoke"])
        err = Exception("rate limit")
        err.status_code = 429
        llm.invoke.side_effect = err

        sleep_times: list[float] = []
        with patch("sema.llm_client.time.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda t: sleep_times.append(t)
            client = LLMClient(
                llm, retry_max_attempts=3, retry_base_delay=2.0,
                rate_limit_base_delay=10.0, retry_jitter=0.0,
            )
            with pytest.raises(LLMStageError):
                client.invoke(
                    "p", TableSummary, table_ref="r", stage_name="s",
                )

        assert len(sleep_times) == 2
        assert sleep_times[0] >= 10.0
        assert sleep_times[1] >= 30.0
