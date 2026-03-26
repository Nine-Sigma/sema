from __future__ import annotations

import logging
from typing import Any, Callable, Final

from sema.graph.loader import GraphLoader

logger = logging.getLogger(__name__)

INDEX_CONFIGS: Final[tuple[tuple[str, str], ...]] = (
    ("entity_embeddings", "Entity"),
    ("property_embeddings", "Property"),
    ("term_embeddings", "Term"),
    ("synonym_embeddings", "Synonym"),
    ("metric_embeddings", "Metric"),
    ("transformation_embeddings", "Transformation"),
)

EMBEDDING_KEY_MAP: Final[dict[str, str | tuple[str, ...]]] = {
    "Entity": "name",
    "Property": ("name", "entity_name"),
    "Term": "code",
    "Synonym": "text",
    "Metric": "name",
    "Transformation": "name",
}


def build_embedding_text(node_type: str, **kwargs: Any) -> str:
    """Build text for embedding from node properties."""
    if node_type == "entity":
        parts = [kwargs.get("name", "")]
        if kwargs.get("description"):
            parts.append(kwargs["description"])
        return " - ".join(parts)

    elif node_type == "property":
        parts = [kwargs.get("name", "")]
        if kwargs.get("description"):
            parts.append(kwargs["description"])
        return " - ".join(parts)

    elif node_type == "term":
        return kwargs.get("label", kwargs.get("code", ""))  # type: ignore[no-any-return]

    elif node_type == "synonym":
        return kwargs.get("text", "")  # type: ignore[no-any-return]

    elif node_type == "metric":
        parts = [kwargs.get("name", "")]
        if kwargs.get("description"):
            parts.append(kwargs["description"])
        return " - ".join(parts)

    elif node_type == "transformation":
        return kwargs.get("name", "")  # type: ignore[no-any-return]

    return kwargs.get("name", kwargs.get("text", ""))  # type: ignore[no-any-return]


class EmbeddingEngine:
    """Compute and store embeddings for graph nodes."""

    def __init__(self, model: Any = None, loader: GraphLoader | None = None) -> None:
        self._model = model
        self._loader = loader

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        if hasattr(self._model, "embed_documents"):
            return self._model.embed_documents(texts)  # type: ignore[no-any-return]
        return self._model.encode(texts)  # type: ignore[no-any-return]

    def embed_and_store(
        self,
        label: str,
        match_prop: str,
        items: list[dict[str, Any]],
        text_fn: Callable[[dict[str, Any]], str],
    ) -> None:
        """Embed items and store embeddings on Neo4j nodes."""
        if not items:
            return

        texts = [text_fn(item) for item in items]
        embeddings = self.embed_batch(texts)

        assert self._loader is not None
        for item, embedding in zip(items, embeddings):
            match_value = item[match_prop]
            self._loader.set_embedding(
                label=label,
                match_prop=match_prop,
                match_value=match_value,
                embedding=list(embedding),
            )

    def create_all_indexes(self, dimensions: int = 1536) -> None:
        """Create vector indexes for all embeddable node types."""
        assert self._loader is not None
        for index_name, label in INDEX_CONFIGS:
            self._loader.create_vector_index(
                index_name, label, dimensions,
            )
