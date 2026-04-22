"""Smoke test: ChatDatabricks / DatabricksEmbeddings import + attribute contract.

No network call is issued. Purpose: any breaking upgrade of
`databricks-langchain` fails the import or the attribute check in CI.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_chat_databricks_exposes_invoke_and_structured_output() -> None:
    from databricks_langchain import ChatDatabricks
    chat = ChatDatabricks(endpoint="does-not-exist-smoke")
    assert hasattr(chat, "invoke")
    assert hasattr(chat, "with_structured_output")


def test_databricks_embeddings_exposes_embed_methods() -> None:
    from databricks_langchain import DatabricksEmbeddings
    emb = DatabricksEmbeddings(endpoint="does-not-exist-smoke")
    assert hasattr(emb, "embed_query")
    assert hasattr(emb, "embed_documents")
