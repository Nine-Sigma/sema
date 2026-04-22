"""Helpers for LLMClient: token accounting, key normalization, structured-output support detection."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from sema.models.protocols import JsonValue, LLMProtocol


def approx_tokens(char_count: int) -> int:
    """Rough English token estimate at ~4 chars per token."""
    return max(0, char_count // 4)


def try_usage_tokens(response: Any) -> tuple[int, int] | None:
    """Extract (prompt, completion) tokens from a langchain response."""
    usage = getattr(response, "usage_metadata", None) or getattr(
        response, "response_metadata", None,
    )
    if isinstance(usage, dict):
        prompt = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("token_usage", {}).get("prompt_tokens")
        )
        completion = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("token_usage", {}).get("completion_tokens")
        )
        if prompt is not None and completion is not None:
            return int(prompt), int(completion)
    return None


def normalize_keys(obj: JsonValue) -> JsonValue:
    if isinstance(obj, dict):
        return {k.lower(): normalize_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_keys(item) for item in obj]
    return obj


class _StructuredProbe(BaseModel):
    ok: bool = True


def probe_structured_output(llm: LLMProtocol) -> bool:
    try:
        llm.with_structured_output(_StructuredProbe).invoke("Respond with ok=true")
        return True
    except Exception:
        return False


def resolve_structured_support(
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
    return probe_structured_output(llm)
