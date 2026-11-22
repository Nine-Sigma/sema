"""Term embedding identity includes ``model_role`` (design 5.3 R1, 7.1 check 8).

A SOURCE term and a TARGET term sharing ``vocabulary_name`` and ``code`` must
not share an embedding write: only the SOURCE vector may change.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.engine.embeddings import EMBEDDING_KEY_MAP
from sema.graph.embedding_match import EMBEDDING_MATCH_KEYS
from sema.graph.loader import GraphLoader
from sema.pipeline.orchestrate_utils import _embed_label_nodes

pytestmark = pytest.mark.unit

TARGET_VECTOR = [9.0, 9.0]
NEW_VECTOR = [0.5, 0.25]
_MATCH_RE = re.compile(r"MATCH \(n:(\w+) \{([^}]*)\}\)")


class InMemorySession:
    """Evaluates ``MATCH (n:Label {k: $m_k, ...}) SET n.embedding ...`` in memory."""

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self._nodes = nodes

    def __enter__(self) -> InMemorySession:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def run(self, query: str, parameters: dict[str, Any] | None = None,
            **kwargs: Any) -> list[Any]:
        params = {**(parameters or {}), **kwargs}
        found = _MATCH_RE.search(query)
        if found is None or "SET n.embedding" not in query:
            raise AssertionError(f"unexpected query: {query}")
        label, pattern = found.groups()
        keys = [part.split(":")[0].strip() for part in pattern.split(",")]
        for node in self._nodes:
            if node["_label"] == label and all(
                node.get(key) == params[f"m_{key}"] for key in keys
            ):
                node["embedding"] = params["embedding"]
        return []


class InMemoryDriver:
    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self.nodes = nodes

    def session(self, **_: Any) -> InMemorySession:
        return InMemorySession(self.nodes)


def _term(role: str, embedding: list[float] | None) -> dict[str, Any]:
    return {
        "_label": "Term", "vocabulary_name": "SYN_VOCAB", "code": "C1",
        "label": "Synthetic code one", "model_role": role, "embedding": embedding,
    }


def _graph() -> tuple[InMemoryDriver, dict[str, Any], dict[str, Any]]:
    source, target = _term("SOURCE", None), _term("TARGET", list(TARGET_VECTOR))
    return InMemoryDriver([source, target]), source, target


def test_term_match_key_appends_model_role() -> None:
    assert EMBEDDING_MATCH_KEYS["Term"] == ("vocabulary_name", "code", "model_role")


def test_set_node_embedding_on_source_term_leaves_target_vector() -> None:
    driver, source, target = _graph()

    GraphLoader(driver).set_node_embedding(
        label="Term",
        match={"vocabulary_name": "SYN_VOCAB", "code": "C1", "model_role": "SOURCE"},
        embedding=list(NEW_VECTOR),
    )

    assert source["embedding"] == NEW_VECTOR
    assert target["embedding"] == TARGET_VECTOR


def test_embedding_loop_write_for_source_term_leaves_target_vector() -> None:
    driver, source, target = _graph()
    engine = MagicMock()
    engine.embed_batch.side_effect = lambda texts: [list(NEW_VECTOR) for _ in texts]
    source_node = {k: v for k, v in source.items() if k != "_label"}

    _embed_label_nodes(
        engine, GraphLoader(driver), "Term", EMBEDDING_KEY_MAP["Term"], [source_node],
    )

    assert source["embedding"] == NEW_VECTOR
    assert target["embedding"] == TARGET_VECTOR
