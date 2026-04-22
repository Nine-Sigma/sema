"""Shared LLM client with standardized response handling.

All LLM interactions in the pipeline go through LLMClient, which owns
response parsing, validation, normalization, retry, and fallback logic.
Engines declare a Pydantic response schema and call invoke() — they never
parse JSON or handle model-specific quirks directly.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Final, TypeVar

from pydantic import BaseModel, ConfigDict

from sema.llm_client_utils import (
    approx_tokens,
    normalize_keys,
    resolve_structured_support,
    try_usage_tokens,
)
from sema.log import logger
from sema.models.protocols import LLMProtocol

T = TypeVar("T", bound=BaseModel)


@dataclass
class InvocationStats:
    """Captured stats from the most recent LLMClient.invoke() call."""

    duration_ns: int = 0
    prompt_chars: int = 0
    response_chars: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class TableSummary(BaseModel):
    """Lightweight entity-level summary for the table summary pass."""
    model_config = ConfigDict(populate_by_name=True)

    entity_name: str
    entity_description: str | None = None
    synonyms: list[str] | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.synonyms is None:
            self.synonyms = []


class VocabularyDetection(BaseModel):
    """Response schema for vocabulary detection calls."""
    model_config = ConfigDict(populate_by_name=True)

    vocabulary: str
    confidence: float = 0.6


class SynonymExpansion(BaseModel):
    """Response schema for synonym expansion calls."""
    model_config = ConfigDict(populate_by_name=True)

    synonyms: list[dict[str, Any]] = []


class LLMStageError(Exception):
    """Raised when all LLM fallback steps are exhausted for a call."""

    def __init__(
        self,
        table_ref: str,
        stage_name: str,
        step_errors: list[tuple[str, Exception]],
    ):
        self.table_ref = table_ref
        self.stage_name = stage_name
        self.step_errors = step_errors
        steps = "; ".join(f"{name}: {err}" for name, err in step_errors)
        super().__init__(
            f"LLM failed for {table_ref} at stage '{stage_name}' — "
            f"all fallback steps exhausted: [{steps}]"
        )


TRANSIENT_STATUS_CODES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})
NON_RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({401, 403})


def _is_transient_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status is not None:
        return int(status) in TRANSIENT_STATUS_CODES
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return True
    msg = str(exc).lower()
    if any(kw in msg for kw in ("rate limit", "timeout", "429", "too many requests")):
        return True
    return False


def parse_llm_response(raw: str, schema: type[T]) -> T:
    """Universal fallback parser for raw LLM text responses."""
    text = raw.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        return schema.model_validate(normalize_keys(parsed))
    except (json.JSONDecodeError, Exception):
        pass

    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        try:
            parsed = json.loads(match.group(1))
            return schema.model_validate(normalize_keys(parsed))
        except (json.JSONDecodeError, Exception):
            pass

    try:
        parsed = json.loads(text) if isinstance(text, str) else text
    except json.JSONDecodeError:
        if match:
            parsed = json.loads(match.group(1))
        else:
            raise ValueError(f"No JSON found in LLM response: {text[:200]}")

    normalized = normalize_keys(parsed)
    if isinstance(normalized, dict):
        for wrapper_key in ("result", "data", "response"):
            if wrapper_key in normalized and isinstance(normalized[wrapper_key], dict):
                try:
                    return schema.model_validate(normalized[wrapper_key])
                except Exception:
                    continue

    raise ValueError(f"Could not parse LLM response into {schema.__name__}: {text[:200]}")


class LLMClient:
    """Standardized LLM interaction wrapper.

    Each worker thread should own its own LLMClient instance.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        retry_max_attempts: int = 3,
        retry_base_delay: float = 2.0,
        retry_multiplier: float = 2.0,
        retry_jitter: float = 0.5,
        use_structured_output: str = "auto",
        circuit_breaker: Any | None = None,
    ):
        self._llm = llm
        self._retry_max_attempts = retry_max_attempts
        self._retry_base_delay = retry_base_delay
        self._retry_multiplier = retry_multiplier
        self._retry_jitter = retry_jitter
        self._circuit_breaker = circuit_breaker

        self._supports_structured = resolve_structured_support(
            llm, use_structured_output,
        )
        self.last_stats: InvocationStats = InvocationStats()
        self._last_response: Any = None

        if self._supports_structured:
            logger.info("LLMClient: using native structured output")
        else:
            has_method = hasattr(llm, "with_structured_output")
            reason = "user-disabled" if has_method else "not supported"
            logger.info(f"LLMClient: using fallback parser ({reason})")

    def invoke(
        self,
        prompt: str,
        schema: type[T],
        *,
        table_ref: str = "",
        stage_name: str = "",
        simplified_prompt: str | None = None,
    ) -> T:
        """Invoke the LLM with a three-step fallback chain."""
        if self._circuit_breaker is not None:
            self._circuit_breaker.check()

        start = time.monotonic_ns()
        self._last_response = None
        try:
            result = self._invoke_fallback_chain(
                prompt, schema, table_ref, stage_name, simplified_prompt,
            )
        except LLMStageError:
            if self._circuit_breaker is not None:
                self._circuit_breaker.record_failure()
            self._record_stats(start, prompt, "")
            raise

        if self._circuit_breaker is not None:
            self._circuit_breaker.record_success()
        response_text = self._response_text_for_stats(result)
        self._record_stats(start, prompt, response_text)
        return result

    def _response_text_for_stats(self, result: Any) -> str:
        if hasattr(result, "model_dump_json"):
            return str(result.model_dump_json())
        return str(result)

    def _record_stats(
        self, start_ns: int, prompt: str, response_text: str,
    ) -> None:
        duration = time.monotonic_ns() - start_ns
        usage = try_usage_tokens(self._last_response)
        if usage is not None:
            prompt_tokens, completion_tokens = usage
        else:
            prompt_tokens = approx_tokens(len(prompt))
            completion_tokens = approx_tokens(len(response_text))
        self.last_stats = InvocationStats(
            duration_ns=duration,
            prompt_chars=len(prompt),
            response_chars=len(response_text),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _invoke_fallback_chain(
        self,
        prompt: str,
        schema: type[T],
        table_ref: str,
        stage_name: str,
        simplified_prompt: str | None,
    ) -> T:
        step_errors: list[tuple[str, Exception]] = []

        if self._supports_structured:
            try:
                return self._invoke_structured(prompt, schema)
            except Exception as e:
                step_errors.append(("structured_output", e))
                logger.debug(f"Structured output failed for {table_ref}: {e}")

        try:
            raw = self._invoke_with_retry(
                lambda: self._raw_invoke(prompt),
                step_name="plain_invoke",
            )
            return parse_llm_response(raw, schema)
        except Exception as e:
            step_errors.append(("plain_invoke", e))
            logger.debug(f"Plain invoke failed for {table_ref}: {e}")

        if simplified_prompt:
            try:
                raw = self._invoke_with_retry(
                    lambda: self._raw_invoke(simplified_prompt),
                    step_name="simplified_prompt",
                )
                return parse_llm_response(raw, schema)
            except Exception as e:
                step_errors.append(("simplified_prompt", e))
                logger.debug(f"Simplified prompt failed for {table_ref}: {e}")

        raise LLMStageError(
            table_ref=table_ref,
            stage_name=stage_name,
            step_errors=step_errors,
        )

    def _invoke_structured(self, prompt: str, schema: type[T]) -> T:
        """Run native structured output and capture raw response for stats.

        Uses `with_structured_output(schema, include_raw=True)` so the raw
        `AIMessage` is available for `try_usage_tokens`. Parse failures
        (returned as `{"parsed": None, "parsing_error": <exc>}`) are raised
        so the plain-invoke fallback path engages.
        """
        def _call() -> Any:
            return self._llm.with_structured_output(
                schema, include_raw=True,
            ).invoke(prompt)

        result = self._invoke_with_retry(_call, step_name="structured_output")
        if isinstance(result, dict):
            return self._unpack_structured_result(result)  # type: ignore[return-value]
        self._last_response = result
        return result  # type: ignore[no-any-return]

    def _unpack_structured_result(self, result: dict[str, Any]) -> BaseModel:
        raw = result.get("raw")
        parsed = result.get("parsed")
        parsing_error = result.get("parsing_error")
        if raw is not None:
            self._last_response = raw
            if try_usage_tokens(raw) is None:
                logger.debug("Structured-output raw response lacked usage_metadata")
        if parsing_error is not None:
            raise parsing_error if isinstance(
                parsing_error, BaseException,
            ) else ValueError(str(parsing_error))
        if parsed is None:
            raise ValueError(
                "Structured output returned parsed=None without a parsing_error",
            )
        return parsed  # type: ignore[no-any-return]

    def _raw_invoke(self, prompt: str) -> str:
        response = self._llm.invoke(prompt)
        self._last_response = response
        return response.content if hasattr(response, "content") else str(response)

    def _invoke_with_retry(self, fn: Callable[..., Any], step_name: str = "") -> Any:
        last_error: Exception | None = None
        for attempt in range(self._retry_max_attempts):
            try:
                return fn()
            except Exception as e:
                last_error = e
                if not _is_transient_error(e) or attempt == self._retry_max_attempts - 1:
                    raise
                delay = self._retry_base_delay * (self._retry_multiplier ** attempt)
                jitter = random.uniform(-self._retry_jitter, self._retry_jitter)
                sleep_time = max(0, delay + jitter)
                logger.debug(
                    f"Retry {attempt + 1}/{self._retry_max_attempts} for {step_name} "
                    f"after {sleep_time:.1f}s: {e}"
                )
                time.sleep(sleep_time)
        raise last_error  # type: ignore[misc]
