"""S0-08: OpenRouter embedding models come from /api/v1/embeddings/models."""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.unit

from sema.cli_factories_utils import (
    OPENROUTER_EMBEDDINGS_CATALOG_URL,
    describe_unknown_embedding_model,
    list_openrouter_embedding_models,
)

EMBEDDINGS_ROUTE = "/api/v1/embeddings/models"
CHAT_ROUTE = "/api/v1/models"


def _transport(seen: list[httpx.Request], status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == EMBEDDINGS_ROUTE:
            return httpx.Response(status, json={"data": [
                {"id": "baai/bge-large-en-v1.5", "context_length": 512},
                {"id": "google/gemini-embedding-2", "context_length": 8192},
            ]})
        # The chat catalog: hundreds of rows, zero embedding models.
        return httpx.Response(200, json={"data": [{"id": "openai/gpt-4o"}]})
    return httpx.MockTransport(handler)


def test_catalog_url_is_the_embeddings_route():
    assert OPENROUTER_EMBEDDINGS_CATALOG_URL.endswith(EMBEDDINGS_ROUTE)


def test_reads_embeddings_catalog_never_chat_catalog():
    seen: list[httpx.Request] = []
    models = list_openrouter_embedding_models("k", transport=_transport(seen))
    assert [r.url.path for r in seen] == [EMBEDDINGS_ROUTE]
    assert CHAT_ROUTE not in {r.url.path for r in seen}
    assert models
    assert "baai/bge-large-en-v1.5" in models
    assert seen[0].headers["authorization"] == "Bearer k"


def test_empty_key_sends_no_auth_header():
    seen: list[httpx.Request] = []
    list_openrouter_embedding_models("", transport=_transport(seen))
    assert "authorization" not in seen[0].headers


def test_unknown_model_message_names_catalog_candidates():
    seen: list[httpx.Request] = []
    msg = describe_unknown_embedding_model(
        "nope/model", "k", transport=_transport(seen),
    )
    assert "nope/model" in msg
    assert "baai/bge-large-en-v1.5" in msg
    assert "google/gemini-embedding-2" in msg
    assert EMBEDDINGS_ROUTE in msg
    assert [r.url.path for r in seen] == [EMBEDDINGS_ROUTE]


def test_catalog_failure_still_names_route_without_raising():
    seen: list[httpx.Request] = []
    msg = describe_unknown_embedding_model(
        "nope/model", "k", transport=_transport(seen, status=503),
    )
    assert "nope/model" in msg
    assert EMBEDDINGS_ROUTE in msg
    assert "baai/bge-large-en-v1.5" not in msg
