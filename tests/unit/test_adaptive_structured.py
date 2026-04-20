from __future__ import annotations

import inspect
from unittest.mock import MagicMock, PropertyMock

import pytest
from pydantic import BaseModel

pytestmark = pytest.mark.unit

from sema.llm_client import LLMClient


class _TrivialSchema(BaseModel):
    name: str = ""


class TestTryFirstProbeAtInitEnablesStructured:
    def test_probe_succeeds_sets_supports_structured_true(self) -> None:
        llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.invoke.return_value = _TrivialSchema(name="probe")
        llm.with_structured_output.return_value = structured_llm

        client = LLMClient(llm, retry_max_attempts=1, use_structured_output="auto")
        assert client._supports_structured is True


class TestTryFirstProbeAtInitDisablesOnFailure:
    def test_probe_raises_sets_supports_structured_false(self) -> None:
        llm = MagicMock()
        llm.with_structured_output.side_effect = Exception("not supported")

        client = LLMClient(llm, retry_max_attempts=1, use_structured_output="auto")
        assert client._supports_structured is False


class TestNoModelNameBlacklist:
    def test_no_reference_to_blacklisted_model_names_in_init(self) -> None:
        source = inspect.getsource(LLMClient.__init__)
        for keyword in ("deepseek", "mistral", "qwen"):
            assert keyword not in source.lower(), (
                f"Found blacklisted model name '{keyword}' in LLMClient.__init__"
            )


class TestUserOverrideTrue:
    def test_structured_always_used_no_probe(self) -> None:
        llm = MagicMock()
        llm.with_structured_output.return_value = MagicMock()

        client = LLMClient(llm, retry_max_attempts=1, use_structured_output="true")
        assert client._supports_structured is True
        llm.with_structured_output.assert_not_called()


class TestUserOverrideFalse:
    def test_structured_never_used(self) -> None:
        llm = MagicMock()
        llm.with_structured_output.return_value = MagicMock()

        client = LLMClient(llm, retry_max_attempts=1, use_structured_output="false")
        assert client._supports_structured is False


class TestUserOverrideAuto:
    def test_auto_triggers_probe(self) -> None:
        llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.invoke.return_value = _TrivialSchema(name="probe")
        llm.with_structured_output.return_value = structured_llm

        client = LLMClient(llm, retry_max_attempts=1, use_structured_output="auto")
        assert client._supports_structured is True
        llm.with_structured_output.assert_called_once()


