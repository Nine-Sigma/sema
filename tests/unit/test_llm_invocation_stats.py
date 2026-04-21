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
