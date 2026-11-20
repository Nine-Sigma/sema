"""Unit tests for sema.cli_factories."""

from __future__ import annotations

import sys
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


class TestGetEmbedderGateway:
    """S0-07: OpenAI-compatible gateways get strings, not tiktoken id arrays."""

    def _call(self, **overrides):
        from sema.cli_factories import _get_embedder
        config = EmbeddingConfig(api_key="test-key", **overrides)
        return _get_embedder(config)

    @patch("openai.resources.embeddings.Embeddings.create")
    def test_openrouter_request_carries_input_as_strings(self, mock_create):
        mock_create.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
        }
        embedder = self._call(
            provider="openrouter", model="baai/bge-large-en-v1.5",
        )
        vector = embedder.embed_query("hello world")
        kwargs = mock_create.call_args.kwargs
        assert kwargs["input"] == ["hello world"]
        assert kwargs["model"] == "baai/bge-large-en-v1.5"
        assert kwargs["encoding_format"] == "float"
        assert vector == [0.1, 0.2, 0.3]

    @patch("langchain_openai.OpenAIEmbeddings")
    def test_openrouter_disables_ctx_length_check(self, mock_cls):
        mock_cls.return_value = MagicMock()
        self._call(provider="openrouter", model="baai/bge-large-en-v1.5")
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["check_embedding_ctx_length"] is False
        assert kwargs["model_kwargs"] == {"encoding_format": "float"}
        assert kwargs["base_url"] == "https://openrouter.ai/api/v1"

    @patch("langchain_openai.OpenAIEmbeddings")
    def test_custom_disables_ctx_length_check(self, mock_cls):
        mock_cls.return_value = MagicMock()
        self._call(
            provider="custom", model="local-embed",
            base_url="http://localhost:8000/v1",
        )
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["check_embedding_ctx_length"] is False
        assert kwargs["model_kwargs"] == {"encoding_format": "float"}
        assert kwargs["base_url"] == "http://localhost:8000/v1"

    @patch("langchain_openai.OpenAIEmbeddings")
    def test_openai_kwargs_unchanged(self, mock_cls):
        mock_cls.return_value = MagicMock()
        self._call(provider="openai", model="text-embedding-3-small")
        kwargs = mock_cls.call_args.kwargs
        assert kwargs == {
            "model": "text-embedding-3-small",
            "api_key": "test-key",
            "base_url": None,
        }

    @patch("langchain_openai.OpenAIEmbeddings")
    def test_openrouter_and_custom_are_wrapped_for_truncation(self, mock_cls):
        from sema.cli_factories_utils import TruncatingEmbedder
        mock_cls.return_value = MagicMock()
        openrouter = self._call(
            provider="openrouter", model="baai/bge-large-en-v1.5",
        )
        custom = self._call(
            provider="custom", model="x", base_url="http://localhost:8000/v1",
        )
        assert isinstance(openrouter, TruncatingEmbedder)
        assert isinstance(custom, TruncatingEmbedder)
        assert openrouter.context_tokens == 512

    @patch("langchain_openai.OpenAIEmbeddings")
    def test_openai_is_not_wrapped(self, mock_cls):
        inner = MagicMock()
        mock_cls.return_value = inner
        assert self._call(provider="openai", model="text-embedding-3-small") is inner


class TestGetEmbedderSentenceTransformers:
    """S0-03: offline provider resolves with no Databricks import."""

    def _call(self, **overrides):
        from sema.cli_factories import _get_embedder
        config = EmbeddingConfig(api_key="test-key", **overrides)
        return _get_embedder(config)

    @patch("sentence_transformers.SentenceTransformer")
    def test_resolves_without_databricks(self, mock_cls, monkeypatch, tmp_path):
        monkeypatch.delenv("DATABRICKS_HOST", raising=False)
        monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
        mock_cls.return_value = MagicMock()
        blocked = {
            "databricks": None, "databricks.sdk": None,
            "databricks_langchain": None,
        }
        with patch.dict(sys.modules, blocked):
            self._call(
                provider="sentence-transformers", model="all-MiniLM-L6-v2",
                cache_dir=str(tmp_path),
            )
        args, kwargs = mock_cls.call_args
        assert args == ("all-MiniLM-L6-v2",)
        assert kwargs == {"cache_folder": str(tmp_path), "local_files_only": True}

    @patch("sentence_transformers.SentenceTransformer")
    def test_env_var_route_reaches_factory(self, mock_cls, monkeypatch, tmp_path):
        from sema.cli_factories import _get_embedder
        monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence-transformers")
        monkeypatch.setenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(tmp_path))
        mock_cls.return_value = MagicMock()
        _get_embedder(EmbeddingConfig())
        assert mock_cls.call_args.kwargs["cache_folder"] == str(tmp_path)

    @patch("sentence_transformers.SentenceTransformer")
    def test_cache_dir_tilde_is_expanded(self, mock_cls):
        mock_cls.return_value = MagicMock()
        self._call(
            provider="sentence-transformers", model="all-MiniLM-L6-v2",
            cache_dir="~/hf-cache",
        )
        folder = mock_cls.call_args.kwargs["cache_folder"]
        assert "~" not in folder and folder.endswith("/hf-cache")

    @patch("sentence_transformers.SentenceTransformer")
    def test_no_cache_dir_keeps_default_call(self, mock_cls, monkeypatch):
        monkeypatch.delenv("EMBEDDING_CACHE_DIR", raising=False)
        mock_cls.return_value = MagicMock()
        self._call(provider="sentence-transformers", model="all-MiniLM-L6-v2")
        mock_cls.assert_called_once_with("all-MiniLM-L6-v2")

    @patch("sentence_transformers.SentenceTransformer")
    def test_empty_cache_dir_means_none(self, mock_cls):
        mock_cls.return_value = MagicMock()
        self._call(
            provider="sentence-transformers", model="all-MiniLM-L6-v2",
            cache_dir="",
        )
        mock_cls.assert_called_once_with("all-MiniLM-L6-v2")

    @patch("sentence_transformers.SentenceTransformer")
    def test_cache_miss_names_prefetch_command(self, mock_cls, tmp_path):
        from sema.cli_factories_utils import EmbeddingCacheMissError
        mock_cls.side_effect = OSError(
            "Cannot find the requested files in the disk cache",
        )
        with pytest.raises(EmbeddingCacheMissError) as exc_info:
            self._call(
                provider="sentence-transformers", model="all-MiniLM-L6-v2",
                cache_dir=str(tmp_path),
            )
        msg = str(exc_info.value)
        expected = (
            "hf download sentence-transformers/all-MiniLM-L6-v2 "
            f"--cache-dir {tmp_path}"
        )
        assert expected in msg
        assert "Cannot find the requested files" in msg
