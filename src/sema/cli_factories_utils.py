"""Helpers behind `cli_factories._get_embedder` for gateway and offline providers.

Three concerns live here so the factory stays thin:

* `TruncatingEmbedder` — sema's own length boundary in front of an
  OpenAI-compatible gateway. With langchain's tiktoken check off (G-23b) no
  client-side chunking happens, so over-limit text would otherwise reach the
  gateway whole. Truncation here is a recorded, bounded loss.
* The OpenRouter embeddings catalog (G-23c): `/api/v1/embeddings/models`,
  never `/api/v1/models`, and only consulted on the error path.
* The sentence-transformers cache-miss message naming the prefetch step.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sema.log import logger

if TYPE_CHECKING:
    import httpx

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_EMBEDDINGS_CATALOG_URL = f"{OPENROUTER_BASE_URL}/embeddings/models"
CATALOG_TIMEOUT_SECONDS = 5.0

CACHE_DIR_ENV_VAR = "EMBEDDING_CACHE_DIR"
SENTENCE_TRANSFORMERS_HUB_ORG = "sentence-transformers"

# `context_length` as read from the OpenRouter embeddings catalog on
# 2026-09-04. A dated observation, not a constant: unlisted models fall back
# to FALLBACK_EMBEDDING_CONTEXT_TOKENS, which is the smallest context seen.
OPENROUTER_EMBEDDING_CONTEXT_TOKENS: dict[str, int] = {
    "baai/bge-base-en-v1.5": 512,
    "baai/bge-large-en-v1.5": 512,
    "baai/bge-m3": 8194,
    "google/gemini-embedding-2": 8192,
    "intfloat/e5-base-v2": 512,
    "intfloat/e5-large-v2": 512,
    "intfloat/multilingual-e5-large": 512,
    "mistralai/mistral-embed-2312": 8192,
    "openai/text-embedding-3-large": 8192,
    "openai/text-embedding-3-small": 8192,
    "openai/text-embedding-ada-002": 8192,
    "sentence-transformers/all-minilm-l6-v2": 512,
    "sentence-transformers/all-minilm-l12-v2": 512,
    "sentence-transformers/all-mpnet-base-v2": 512,
    "thenlper/gte-base": 512,
    "thenlper/gte-large": 512,
}
FALLBACK_EMBEDDING_CONTEXT_TOKENS = 512
# Character-based approximation of the token budget. Sema has no tokenizer
# for gateway models, so the cut is chars = tokens * this. Three is on the
# low side for English prose (~4) so the cut lands under the limit for prose;
# identifier-heavy text can still tokenize denser than this.
APPROX_CHARS_PER_TOKEN = 3
REJECTED_MODEL_STATUSES: frozenset[int] = frozenset({400, 404})


class EmbeddingCacheMissError(RuntimeError):
    """The sentence-transformers model is not in the local cache."""


def embedding_context_tokens(model: str) -> int:
    return OPENROUTER_EMBEDDING_CONTEXT_TOKENS.get(
        model.lower(), FALLBACK_EMBEDDING_CONTEXT_TOKENS,
    )


class TruncatingEmbedder:
    """Truncate each text to a declared budget before the gateway sees it.

    Exposes exactly the langchain `embed_documents` / `embed_query` contract
    that `EmbeddingEngine`, `RetrievalEngine` and `probe_embedder_dimension`
    duck-type against; deliberately no `encode` and no
    `get_sentence_embedding_dimension`, so callers branch as they do today.
    Truncations are counted on `truncated_count` and warned once per run.
    """

    def __init__(
        self,
        inner: Any,
        *,
        model: str,
        context_tokens: int,
        describe_rejected_model: Callable[[str], str] | None = None,
    ) -> None:
        self._inner = inner
        self.model = model
        self.context_tokens = context_tokens
        self.char_budget = context_tokens * APPROX_CHARS_PER_TOKEN
        self._describe_rejected_model = describe_rejected_model
        self.truncated_count = 0
        self._warned = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        bounded = [self._bound(text) for text in texts]
        result: list[list[float]] = self._call(
            self._inner.embed_documents, bounded,
        )
        return result

    def embed_query(self, text: str) -> list[float]:
        result: list[float] = self._call(self._inner.embed_query, self._bound(text))
        return result

    def _bound(self, text: str) -> str:
        if len(text) <= self.char_budget:
            return text
        self.truncated_count += 1
        self._warn_once(len(text))
        return text[: self.char_budget]

    def _warn_once(self, observed_chars: int) -> None:
        if self._warned:
            return
        self._warned = True
        logger.warning(
            "Embedding input of {} chars exceeds the ~{}-token context of "
            "'{}'; truncating to {} chars ({} chars/token approximation). "
            "Further truncations this run are counted silently "
            "(TruncatingEmbedder.truncated_count).",
            observed_chars, self.context_tokens, self.model,
            self.char_budget, APPROX_CHARS_PER_TOKEN,
        )

    def _call(self, fn: Callable[[Any], Any], arg: Any) -> Any:
        try:
            return fn(arg)
        except Exception as exc:
            if self._describe_rejected_model is None or not self._rejects_model(exc):
                raise
            raise ValueError(self._describe_rejected_model(self.model)) from exc

    def _rejects_model(self, exc: Exception) -> bool:
        # Lazy: the openai SDK is only needed once a gateway has answered.
        from openai import APIStatusError
        if not isinstance(exc, APIStatusError):
            return False
        return exc.status_code in REJECTED_MODEL_STATUSES and self.model in str(exc)


def list_openrouter_embedding_models(
    api_key: str = "",
    *,
    transport: httpx.BaseTransport | None = None,
) -> list[str]:
    """Model ids from the embeddings catalog (the chat catalog lists none)."""
    import httpx
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(
        transport=transport, timeout=CATALOG_TIMEOUT_SECONDS, headers=headers,
    ) as client:
        response = client.get(OPENROUTER_EMBEDDINGS_CATALOG_URL)
        response.raise_for_status()
        rows = response.json().get("data", [])
    return [row["id"] for row in rows if isinstance(row, dict) and "id" in row]


def describe_unknown_embedding_model(
    model: str,
    api_key: str = "",
    *,
    transport: httpx.BaseTransport | None = None,
) -> str:
    import httpx
    head = f"OpenRouter rejected embedding model '{model}'."
    try:
        candidates = list_openrouter_embedding_models(api_key, transport=transport)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "Could not read the OpenRouter embeddings catalog {}: {}",
            OPENROUTER_EMBEDDINGS_CATALOG_URL, exc,
        )
        return (
            f"{head} The embeddings catalog "
            f"({OPENROUTER_EMBEDDINGS_CATALOG_URL}) could not be read: {exc}. "
            "Pick a model listed there."
        )
    return (
        f"{head} Models in the OpenRouter embeddings catalog "
        f"({OPENROUTER_EMBEDDINGS_CATALOG_URL}): {', '.join(sorted(candidates))}."
    )


def expand_cache_dir(cache_dir: str | None) -> str | None:
    """`EmbeddingConfig.cache_dir` with `~` expanded; blank means unset."""
    value = (cache_dir or "").strip()
    return os.path.expanduser(value) if value else None


def hub_repo_id(model: str) -> str:
    """Mirror sentence-transformers' own rule for bare model names."""
    if "/" in model:
        return model
    return f"{SENTENCE_TRANSFORMERS_HUB_ORG}/{model}"


def sentence_transformers_prefetch_hint(
    model: str, cache_dir: str | None, cause: BaseException,
) -> str:
    command = f"hf download {hub_repo_id(model)}"
    where = f"set cache_dir / {CACHE_DIR_ENV_VAR} to a warm cache"
    if cache_dir is not None:
        command = f"{command} --cache-dir {cache_dir}"
        where = f"cache_dir={cache_dir}"
    return (
        f"sentence-transformers model '{model}' is not available locally "
        f"({where}). Prefetch it once with: {command}. Underlying error: {cause}"
    )
