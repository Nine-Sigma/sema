"""Unit tests for sema.cli_factories."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from sema.models.config import EmbeddingConfig, LLMConfig


class TestGetLLM:
    def _call(self, **overrides):
        from sema.cli_factories import _get_llm
        config = LLMConfig(api_key="test-key", **overrides)
        return _get_llm(config)

    @patch("langchain_openai.ChatOpenAI")
    def test_openrouter_provider(self, mock_cls):
        mock_cls.return_value = MagicMock()
        result = self._call(provider="openrouter", model="test-model")
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert kwargs["model"] == "test-model"
        assert kwargs["request_timeout"] == 120

    @patch("langchain_anthropic.ChatAnthropic")
    def test_anthropic_provider(self, mock_cls):
        mock_cls.return_value = MagicMock()
        result = self._call(provider="anthropic", model="claude-sonnet-4")
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4"
        assert kwargs["timeout"] == 120.0

    @patch("langchain_openai.ChatOpenAI")
    def test_openai_provider(self, mock_cls):
        mock_cls.return_value = MagicMock()
        result = self._call(provider="openai", model="gpt-4o")
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["request_timeout"] == 120

    @patch("langchain_openai.ChatOpenAI")
    def test_databricks_provider(self, mock_cls):
        mock_cls.return_value = MagicMock()
        result = self._call(
            provider="databricks", model="dbrx",
            base_url="https://my-workspace.databricks.com/serving",
        )
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["base_url"] == "https://my-workspace.databricks.com/serving"

    @patch("langchain_openai.ChatOpenAI")
    def test_custom_provider(self, mock_cls):
        mock_cls.return_value = MagicMock()
        result = self._call(
            provider="custom", model="local",
            base_url="http://localhost:8000/v1",
        )
        mock_cls.assert_called_once()

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            self._call(provider="unknown")

    @patch("langchain_openai.ChatOpenAI")
    def test_custom_timeout(self, mock_cls):
        mock_cls.return_value = MagicMock()
        self._call(provider="openai", request_timeout=300)
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["request_timeout"] == 300


class TestGetEmbedder:
    def _call(self, **overrides):
        from sema.cli_factories import _get_embedder
        config = EmbeddingConfig(api_key="test-key", **overrides)
        return _get_embedder(config)

    @patch("sentence_transformers.SentenceTransformer")
    def test_sentence_transformers_provider(self, mock_cls):
        mock_cls.return_value = MagicMock()
        self._call(provider="sentence-transformers", model="all-MiniLM-L6-v2")
        mock_cls.assert_called_once_with("all-MiniLM-L6-v2")

    @patch("langchain_openai.OpenAIEmbeddings")
    def test_openai_provider(self, mock_cls):
        mock_cls.return_value = MagicMock()
        self._call(provider="openai", model="text-embedding-3-small")
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == "text-embedding-3-small"

    @patch("langchain_openai.OpenAIEmbeddings")
    def test_openrouter_embedding_provider(self, mock_cls):
        mock_cls.return_value = MagicMock()
        self._call(provider="openrouter")
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["base_url"] == "https://openrouter.ai/api/v1"

    def test_unknown_embedding_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            self._call(provider="unknown")
