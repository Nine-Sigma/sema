"""Embedding loop role filter (design 5.3 R1 part 1, 5.9).

Nodes with ``model_role = 'TARGET'`` are neither embedded nor written. SOURCE
nodes, and nodes with no ``model_role`` (a source build's list today), are.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sema.models.config import BuildConfig, EmbeddingConfig
from sema.pipeline.orchestrate_utils import _compute_embeddings

pytestmark = pytest.mark.unit

TARGET_MARK = "tgt_"
NODES: dict[str, list[dict[str, Any]]] = {
    "Entity": [{"name": "Visit", "description": "One clinic visit"}],
    "Property": [
        {"name": "visit_kind", "entity_name": "Visit", "model_role": "SOURCE"},
        {"name": f"{TARGET_MARK}visit_kind", "entity_name": f"{TARGET_MARK}Visit",
         "model_role": "TARGET"},
    ],
    "Term": [
        {"vocabulary_name": "SYN_VOCAB", "code": "C1", "label": "Code one",
         "model_role": "SOURCE"},
        {"vocabulary_name": f"{TARGET_MARK}VOCAB", "code": f"{TARGET_MARK}C1",
         "label": f"{TARGET_MARK}label", "model_role": "TARGET"},
    ],
}


class FakeModel:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[0.1, 0.2] for _ in texts]


def _run() -> tuple[FakeModel, MagicMock]:
    model = FakeModel()
    loader = MagicMock()
    loader.query_nodes_by_label.side_effect = lambda label: [
        dict(node) for node in NODES.get(label, [])
    ]
    config = BuildConfig(
        catalog="syn_cat", schemas=["clinical"],
        embedding=EmbeddingConfig(embeddable_labels=["Entity", "Property", "Term"]),
    )
    with patch("sema.pipeline.orchestrate_utils._get_embedder", return_value=model), \
            patch("sema.graph.vector_index_utils.assert_write_dim_matches"):
        _compute_embeddings(config, loader)
    return model, loader


def _written(loader: MagicMock) -> list[tuple[str, dict[str, str]]]:
    return [
        (c.kwargs["label"], dict(c.kwargs["match"]))
        for c in loader.set_node_embedding.call_args_list
    ]


def test_target_nodes_are_not_embedded() -> None:
    model, _ = _run()

    assert model.texts, "the loop embedded nothing"
    assert not [text for text in model.texts if TARGET_MARK in text]


def test_target_nodes_are_not_written() -> None:
    _, loader = _run()

    written = _written(loader)
    assert not [
        (label, match) for label, match in written
        if any(TARGET_MARK in str(value) for value in match.values())
    ]


def test_source_and_unroled_nodes_are_written_as_today() -> None:
    _, loader = _run()

    labels = sorted(label for label, _ in _written(loader))
    assert labels == ["Entity", "Property", "Term"]
