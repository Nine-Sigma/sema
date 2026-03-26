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
from typing import Any, Callable, Final, TypeVar

from pydantic import BaseModel, ConfigDict

from sema.log import logger
from sema.models.protocols import JsonValue, LLMProtocol

T = TypeVar("T", bound=BaseModel)


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
    """Raised when all LLM fallback steps are exhausted for a call.

    Engines MUST NOT catch this — it propagates to _process_table()
    which converts it to TableResult.failed.
    """

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
    # Parse failures are transient (LLM may succeed on retry)
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return True
    msg = str(exc).lower()
    if any(kw in msg for kw in ("rate limit", "timeout", "429", "too many requests")):
        return True
    return False


def _normalize_keys(obj: JsonValue) -> JsonValue:
    if isinstance(obj, dict):
        return {k.lower(): _normalize_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_keys(item) for item in obj]
    return obj


def parse_llm_response(raw: str, schema: type[T]) -> T:
    """Universal fallback parser for raw LLM text responses.

    Pipeline: strip markdown fences → strip prose → extract JSON block →
    json.loads → normalize keys → schema.model_validate.
    """
    text = raw.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        normalized = _normalize_keys(parsed)
        return schema.model_validate(normalized)
    except (json.JSONDecodeError, Exception):
        pass

    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        try:
            parsed = json.loads(match.group(1))
            normalized = _normalize_keys(parsed)
            return schema.model_validate(normalized)
        except (json.JSONDecodeError, Exception):
            pass

    try:
        parsed = json.loads(text) if isinstance(text, str) else text
    except json.JSONDecodeError:
        if match:
            parsed = json.loads(match.group(1))
        else:
            raise ValueError(f"No JSON found in LLM response: {text[:200]}")

    normalized = _normalize_keys(parsed)
    if isinstance(normalized, dict):
        for wrapper_key in ("result", "data", "response"):
            if wrapper_key in normalized and isinstance(normalized[wrapper_key], dict):
                try:
                    return schema.model_validate(normalized[wrapper_key])
                except Exception:
                    continue

    raise ValueError(f"Could not parse LLM response into {schema.__name__}: {text[:200]}")


class _StructuredProbe(BaseModel):
    ok: bool = True


def _resolve_structured_support(
    llm: LLMProtocol,
    use_structured_output: str,
) -> bool:
    has_method = hasattr(llm, "with_structured_output")
    mode = use_structured_output.lower().strip()

    if mode == "true":
        return has_method
    if mode == "false":
        return False

    if not has_method:
        return False
    return _probe_structured_output(llm)


def _probe_structured_output(llm: LLMProtocol) -> bool:
    try:
        llm.with_structured_output(_StructuredProbe).invoke("Respond with ok=true")
        return True
    except Exception:
        return False


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

        self._supports_structured = _resolve_structured_support(
            llm, use_structured_output,
        )

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
        """Invoke the LLM with a three-step fallback chain.

        1. Native structured output (if supported)
        2. Plain invoke + universal fallback parser
        3. Simplified prompt + universal fallback parser

        Raises LLMStageError if all steps are exhausted.
        """
        if self._circuit_breaker is not None:
            self._circuit_breaker.check()

        try:
            result = self._invoke_fallback_chain(
                prompt, schema, table_ref, stage_name, simplified_prompt,
            )
        except LLMStageError:
            if self._circuit_breaker is not None:
                self._circuit_breaker.record_failure()
            raise

        if self._circuit_breaker is not None:
            self._circuit_breaker.record_success()
        return result

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
                return self._invoke_with_retry(  # type: ignore[no-any-return]
                    lambda: self._llm.with_structured_output(schema).invoke(prompt),
                    step_name="structured_output",
                )
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

    def _raw_invoke(self, prompt: str) -> str:
        response = self._llm.invoke(prompt)
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
