"""Unit tests for sema.cli_factories."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from sema.cli_factories import DatabricksProviderAuthError
from sema.models.config import EmbeddingConfig, LLMConfig


class TestGetLLM:
    def _call(self, **overrides):
        from sema.cli_factories import _get_llm
        config = LLMConfig(api_key="test-key", **overrides)
        return _get_llm(config)

    @patch("langchain_openai.ChatOpenAI")
    def test_openrouter_provider(self, mock_cls):
        mock_cls.return_value = MagicMock()
        self._call(provider="openrouter", model="test-model")
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert kwargs["model"] == "test-model"
        assert kwargs["request_timeout"] == 120

    @patch("langchain_anthropic.ChatAnthropic")
    def test_anthropic_provider(self, mock_cls):
        mock_cls.return_value = MagicMock()
        self._call(provider="anthropic", model="claude-sonnet-4")
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4"
        assert kwargs["timeout"] == 120.0

    @patch("langchain_openai.ChatOpenAI")
    def test_openai_provider(self, mock_cls):
        mock_cls.return_value = MagicMock()
        self._call(provider="openai", model="gpt-4o")
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["request_timeout"] == 120

    @patch("langchain_openai.ChatOpenAI")
    def test_custom_provider(self, mock_cls):
        mock_cls.return_value = MagicMock()
        self._call(
            provider="custom", model="local",
            base_url="http://localhost:8000/v1",
        )
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["base_url"] == "http://localhost:8000/v1"
        assert kwargs["model"] == "local"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            self._call(provider="unknown")

    @patch("langchain_openai.ChatOpenAI")
    def test_custom_timeout(self, mock_cls):
        mock_cls.return_value = MagicMock()
        self._call(provider="openai", request_timeout=300)
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["request_timeout"] == 300


class TestGetLLMDatabricks:
    def _call(self, **overrides):
        from sema.cli_factories import _get_llm
        config = LLMConfig(api_key="test-key", **overrides)
        return _get_llm(config)

    @patch("databricks.sdk.WorkspaceClient")
    @patch("databricks_langchain.ChatDatabricks")
    def test_returns_chat_databricks_with_endpoint(
        self, mock_chat_cls, mock_workspace_cls,
    ):
        mock_chat_cls.return_value = MagicMock()
        mock_workspace_cls.return_value = MagicMock()
        self._call(provider="databricks", model="databricks-llama-4-maverick")
        mock_chat_cls.assert_called_once()
        kwargs = mock_chat_cls.call_args.kwargs
        assert kwargs["endpoint"] == "databricks-llama-4-maverick"
        assert "api_key" not in kwargs
        assert "base_url" not in kwargs
        assert "target_uri" not in kwargs

    @patch("databricks.sdk.WorkspaceClient")
    @patch("databricks_langchain.ChatDatabricks")
    def test_ignores_stale_openrouter_key(
        self, mock_chat_cls, mock_workspace_cls,
    ):
        from sema.cli_factories import _get_llm
        mock_chat_cls.return_value = MagicMock()
        mock_workspace_cls.return_value = MagicMock()
        config = LLMConfig(
            provider="databricks",
            model="databricks-llama-4-maverick",
            api_key="sk-or-v1-stale-openrouter-token",
            base_url="https://openrouter.ai/api/v1",
        )
        _get_llm(config)
        kwargs = mock_chat_cls.call_args.kwargs
        assert "api_key" not in kwargs
        assert "base_url" not in kwargs

    @patch("databricks.sdk.WorkspaceClient")
    def test_raises_databricks_auth_error_when_sdk_fails(
        self, mock_workspace_cls,
    ):
        mock_workspace_cls.side_effect = RuntimeError("no creds")
        with pytest.raises(DatabricksProviderAuthError) as exc_info:
            self._call(
                provider="databricks", model="databricks-llama-4-maverick",
            )
        msg = str(exc_info.value)
        assert "DATABRICKS_HOST" in msg
        assert "DATABRICKS_TOKEN" in msg
        assert "DATABRICKS_CONFIG_PROFILE" in msg

    @patch("databricks.sdk.WorkspaceClient")
    def test_raises_databricks_auth_error_when_current_user_fails(
        self, mock_workspace_cls,
    ):
        wc = MagicMock()
        wc.current_user.me.side_effect = Exception("auth denied")
        mock_workspace_cls.return_value = wc
        with pytest.raises(DatabricksProviderAuthError):
            self._call(
                provider="databricks", model="databricks-llama-4-maverick",
            )

    @pytest.mark.parametrize(
        "model",
        [
            "databricks-gpt-oss-120b",
            "databricks-gpt-oss-20b",
            "databricks-gpt-5-3-codex",
            "databricks-gpt-5-2-codex",
        ],
    )
    def test_unsupported_endpoint_raises(self, model):
        with pytest.raises(ValueError) as exc_info:
            self._call(provider="databricks", model=model)
        assert model in str(exc_info.value)

    @patch("databricks.sdk.WorkspaceClient")
    @patch("databricks_langchain.ChatDatabricks")
    @pytest.mark.parametrize(
        "model",
        ["databricks-llama-4-maverick", "databricks-gemma-3-12b"],
    )
    def test_supported_endpoints_construct(
        self, mock_chat_cls, mock_workspace_cls, model,
    ):
        mock_chat_cls.return_value = MagicMock()
        mock_workspace_cls.return_value = MagicMock()
        self._call(provider="databricks", model=model)
        mock_chat_cls.assert_called_once()
        assert mock_chat_cls.call_args.kwargs["endpoint"] == model


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

    @patch("databricks.sdk.WorkspaceClient")
    @patch("databricks_langchain.DatabricksEmbeddings")
    def test_databricks_provider_returns_embeddings(
        self, mock_emb_cls, mock_workspace_cls,
    ):
        mock_emb_cls.return_value = MagicMock()
        mock_workspace_cls.return_value = MagicMock()
        self._call(provider="databricks", model="databricks-bge-large-en")
        mock_emb_cls.assert_called_once()
        kwargs = mock_emb_cls.call_args.kwargs
        assert kwargs["endpoint"] == "databricks-bge-large-en"
        assert "api_key" not in kwargs
        assert "base_url" not in kwargs
        assert "target_uri" not in kwargs

    @patch("databricks.sdk.WorkspaceClient")
    def test_databricks_embedder_raises_auth_error(
        self, mock_workspace_cls,
    ):
        mock_workspace_cls.side_effect = RuntimeError("no creds")
        with pytest.raises(DatabricksProviderAuthError):
            self._call(provider="databricks", model="databricks-bge-large-en")

    def test_unknown_embedding_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            self._call(provider="unknown")
