"""Change-detection helpers for the embedding phase (finding M).

Embeddings are computed once post-build with no change detection, so a
later study overwriting a shared node's description leaves a stale vector.
These pure helpers let the phase skip nodes whose embedded text is
unchanged: each node stores ``description_hash`` (sha256 of the embedded
text) alongside its vector, and only nodes missing an embedding or whose
hash no longer matches are re-embedded.
"""

from __future__ import annotations

import hashlib
from typing import Any


def description_hash(text: str) -> str:
    """SHA-256 hex digest of the text a node was embedded from."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def needs_reembedding(node: dict[str, Any], text: str) -> bool:
    """True when a node has no embedding or its embedded text changed."""
    if not node.get("embedding"):
        return True
    return node.get("description_hash") != description_hash(text)


def select_stale_nodes(
    nodes: list[dict[str, Any]], texts: list[str],
) -> list[tuple[dict[str, Any], str]]:
    """Pair each node with its text where re-embedding is required."""
    return [
        (node, text)
        for node, text in zip(nodes, texts)
        if needs_reembedding(node, text)
    ]
