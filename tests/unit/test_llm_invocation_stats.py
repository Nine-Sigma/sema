"""Tests for per-invocation LLM stats (latency, estimated tokens)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from sema.llm_client import InvocationStats, LLMClient

pytestmark = pytest.mark.unit


class _Probe(BaseModel):
    value: str = "ok"


def _make_client_with_raw(response_text: str) -> LLMClient:
    llm = MagicMock()
    llm.with_structured_output = None
    del llm.with_structured_output
    llm.invoke = MagicMock(
        return_value=MagicMock(content=response_text),
    )
    return LLMClient(llm=llm, use_structured_output="false")


class TestInvocationStatsCapture:
    def test_defaults_before_any_call(self) -> None:
        client = _make_client_with_raw('{"value": "ok"}')
        stats = client.last_stats
        assert isinstance(stats, InvocationStats)
        assert stats.duration_ns == 0
        assert stats.prompt_chars == 0
        assert stats.response_chars == 0

    def test_captures_prompt_and_response_chars(self) -> None:
        client = _make_client_with_raw('{"value": "ok"}')
        client.invoke("abc" * 10, _Probe)
        stats = client.last_stats
        assert stats.prompt_chars == 30
        assert stats.response_chars == len('{"value":"ok"}')

    def test_captures_positive_duration(self) -> None:
        client = _make_client_with_raw('{"value": "ok"}')
        client.invoke("hi", _Probe)
        assert client.last_stats.duration_ns > 0

    def test_estimates_tokens_from_chars_at_one_quarter(self) -> None:
        client = _make_client_with_raw('{"value": "ok"}')
        client.invoke("a" * 400, _Probe)
        assert client.last_stats.prompt_tokens == 100
        assert client.last_stats.completion_tokens == (
            len('{"value":"ok"}') // 4
        )


class TestInvocationStatsReset:
    def test_each_call_replaces_previous_stats(self) -> None:
        client = _make_client_with_raw('{"value": "ok"}')
        client.invoke("first" * 50, _Probe)
        first_chars = client.last_stats.prompt_chars
        client.invoke("x", _Probe)
        assert client.last_stats.prompt_chars == 1
        assert first_chars != client.last_stats.prompt_chars


def _ai_message(text: str, *, input_tokens: int, output_tokens: int) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    msg.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    return msg


def _make_structured_client(raw_response: MagicMock, parsed: _Probe) -> LLMClient:
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = {
        "raw": raw_response,
        "parsed": parsed,
        "parsing_error": None,
    }
    llm.with_structured_output.return_value = structured
    return LLMClient(llm=llm, use_structured_output="true")


class TestStructuredOutputStats:
    def test_records_real_usage_from_ai_message(self) -> None:
        raw = _ai_message(
            "{}", input_tokens=123, output_tokens=45,
        )
        client = _make_structured_client(raw, _Probe())
        client.invoke("prompt" * 10, _Probe)
        assert client.last_stats.prompt_tokens == 123
        assert client.last_stats.completion_tokens == 45

    def test_falls_back_to_char_estimates_when_no_metadata(self) -> None:
        raw = MagicMock()
        raw.content = "{}"
        del raw.usage_metadata
        del raw.response_metadata
        client = _make_structured_client(raw, _Probe())
        prompt = "a" * 400
        client.invoke(prompt, _Probe)
        assert client.last_stats.prompt_tokens == 100


class TestStructuredOutputParseFailureFallback:
    def test_parse_error_triggers_plain_invoke_fallback(self) -> None:
        raw = _ai_message("{}", input_tokens=10, output_tokens=5)
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke.return_value = {
            "raw": raw,
            "parsed": None,
            "parsing_error": ValueError("could not parse schema"),
        }
        llm.with_structured_output.return_value = structured
        llm.invoke = MagicMock(
            return_value=MagicMock(content='{"value": "fallback"}'),
        )
        client = LLMClient(llm=llm, use_structured_output="true")
        result = client.invoke("prompt", _Probe)
        assert result.value == "fallback"
        assert llm.invoke.called, "plain-invoke fallback did not engage"
