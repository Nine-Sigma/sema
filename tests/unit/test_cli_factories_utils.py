"""Unit tests for sema.cli_factories_utils: the truncation boundary."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

pytestmark = pytest.mark.unit

from sema.cli_factories_utils import (
    APPROX_CHARS_PER_TOKEN,
    FALLBACK_EMBEDDING_CONTEXT_TOKENS,
    TruncatingEmbedder,
    embedding_context_tokens,
    sentence_transformers_prefetch_hint,
)


def _inner() -> MagicMock:
    inner = MagicMock(spec=["embed_documents", "embed_query"])
    inner.embed_documents.side_effect = (
        lambda texts: [[float(len(t))] for t in texts]
    )
    inner.embed_query.side_effect = lambda text: [float(len(text))]
    return inner


def _rejection(status: int, message: str) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/embeddings")
    response = httpx.Response(status, request=request)
    return openai.APIStatusError(message, response=response, body=None)


class TestTruncatingEmbedder:
    def test_short_text_passes_through_unchanged(self):
        inner = _inner()
        wrapper = TruncatingEmbedder(inner, model="m", context_tokens=4)
        with patch("sema.cli_factories_utils.logger") as log:
            assert wrapper.embed_documents(["abc"]) == [[3.0]]
        inner.embed_documents.assert_called_once_with(["abc"])
        assert wrapper.truncated_count == 0
        log.warning.assert_not_called()

    def test_over_limit_text_truncated_counted_logged_once(self):
        inner = _inner()
        wrapper = TruncatingEmbedder(inner, model="m", context_tokens=4)
        budget = 4 * APPROX_CHARS_PER_TOKEN
        with patch("sema.cli_factories_utils.logger") as log:
            wrapper.embed_documents(["x" * 40, "short", "y" * (budget + 1)])
            wrapper.embed_documents(["z" * 100])
        sent = inner.embed_documents.call_args_list[0].args[0]
        assert [len(t) for t in sent] == [budget, 5, budget]
        assert wrapper.truncated_count == 3
        assert log.warning.call_count == 1

    def test_embed_query_truncates(self):
        inner = _inner()
        wrapper = TruncatingEmbedder(inner, model="m", context_tokens=1)
        with patch("sema.cli_factories_utils.logger"):
            vec = wrapper.embed_query("a" * 50)
        assert vec == [float(APPROX_CHARS_PER_TOKEN)]
        assert wrapper.truncated_count == 1

    def test_exposes_only_the_langchain_contract(self):
        wrapper = TruncatingEmbedder(_inner(), model="m", context_tokens=4)
        assert not hasattr(wrapper, "encode")
        assert not hasattr(wrapper, "get_sentence_embedding_dimension")


class TestModelRejection:
    def test_4xx_naming_model_raises_catalog_message(self):
        inner = _inner()
        inner.embed_documents.side_effect = _rejection(
            400, "nope/x is not a valid model ID",
        )
        wrapper = TruncatingEmbedder(
            inner, model="nope/x", context_tokens=512,
            describe_rejected_model=lambda m: f"candidates for {m}: a, b",
        )
        with pytest.raises(ValueError, match="candidates for nope/x: a, b"):
            wrapper.embed_documents(["t"])

    def test_4xx_not_naming_model_reraises_original(self):
        inner = _inner()
        inner.embed_query.side_effect = _rejection(400, "input too long")
        wrapper = TruncatingEmbedder(
            inner, model="nope/x", context_tokens=512,
            describe_rejected_model=lambda m: "unused",
        )
        with pytest.raises(openai.APIStatusError, match="input too long"):
            wrapper.embed_query("t")

    def test_non_gateway_error_passes_through_untouched(self):
        inner = _inner()
        inner.embed_query.side_effect = RuntimeError("nope/x connection reset")
        wrapper = TruncatingEmbedder(
            inner, model="nope/x", context_tokens=512,
            describe_rejected_model=lambda m: "unused",
        )
        with pytest.raises(RuntimeError, match="connection reset"):
            wrapper.embed_query("t")

    def test_without_describer_reraises_original(self):
        inner = _inner()
        inner.embed_query.side_effect = _rejection(404, "nope/x not found")
        wrapper = TruncatingEmbedder(inner, model="nope/x", context_tokens=512)
        with pytest.raises(openai.APIStatusError):
            wrapper.embed_query("t")


class TestEmbeddingContextTokens:
    def test_measured_bge_large_is_512(self):
        assert embedding_context_tokens("baai/bge-large-en-v1.5") == 512

    def test_lookup_is_case_insensitive(self):
        assert embedding_context_tokens("BAAI/bge-large-en-v1.5") == 512

    def test_unknown_model_uses_conservative_fallback(self):
        assert (
            embedding_context_tokens("vendor/never-heard-of-it")
            == FALLBACK_EMBEDDING_CONTEXT_TOKENS
        )


class TestPrefetchHint:
    def test_bare_name_gets_hub_org_and_cache_dir(self):
        hint = sentence_transformers_prefetch_hint(
            "all-MiniLM-L6-v2", "/tmp/hf", OSError("missing"),
        )
        assert (
            "hf download sentence-transformers/all-MiniLM-L6-v2 "
            "--cache-dir /tmp/hf" in hint
        )
        assert "missing" in hint

    def test_qualified_name_kept_and_no_cache_dir_names_env_var(self):
        hint = sentence_transformers_prefetch_hint(
            "BAAI/bge-small-en-v1.5", None, OSError("offline"),
        )
        assert "hf download BAAI/bge-small-en-v1.5" in hint
        assert "--cache-dir" not in hint
        assert "EMBEDDING_CACHE_DIR" in hint
