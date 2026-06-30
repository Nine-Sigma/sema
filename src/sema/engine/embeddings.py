from __future__ import annotations

import logging
from typing import Any, Callable, Final

from sema.graph.embedding_match import EMBEDDING_MATCH_KEYS
from sema.graph.loader import GraphLoader

logger = logging.getLogger(__name__)

# Default index configs — Transformation removed; Alias replaces Synonym.
# Override by passing embeddable_labels to EmbeddingEngine.__init__.
INDEX_CONFIGS: Final[tuple[tuple[str, str], ...]] = (
    ("entity_embedding_index", "Entity"),
    ("property_embedding_index", "Property"),
    ("term_embedding_index", "Term"),
    ("alias_embedding_index", "Alias"),
    ("metric_embedding_index", "Metric"),
)

# Label -> ordered tuple of node properties forming its embedding match key.
# Single source of truth lives in graph.embedding_match so the loader (which
# cannot import this engine module without a cycle) shares the same allowlist.
# Term keys on the composite {vocabulary_name, code} so same-code terms in
# different vocabularies keep distinct embeddings.
EMBEDDING_KEY_MAP: Final[dict[str, tuple[str, ...]]] = dict(
    EMBEDDING_MATCH_KEYS
)


def _build_match(
    label: str, match_keys: tuple[str, ...], node: dict[str, Any],
) -> dict[str, str] | None:
    """Build a composite match dict from a node's properties.

    Returns None (and warns) when the node is missing a required match
    property — e.g. a legacy ``:Term`` written before composite identity
    that has no ``vocabulary_name``. Skipping is deliberate: matching on a
    partial key would clobber the embeddings of unrelated same-code terms.
    """
    match: dict[str, str] = {}
    for prop in match_keys:
        value = node.get(prop)
        if value is None:
            logger.warning(
                "Skipping %s embedding: node missing match property '%s' "
                "(node keys: %s)",
                label, prop, sorted(node),
            )
            return None
        match[prop] = value
    return match


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

    elif node_type in ("synonym", "alias"):
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
    """Compute and store embeddings for graph nodes.

    Pass embeddable_labels to override the default INDEX_CONFIGS set.
    This allows config-driven control over which node types get embedded.
    """

    def __init__(
        self,
        model: Any = None,
        loader: GraphLoader | None = None,
        embeddable_labels: list[str] | None = None,
    ) -> None:
        self._model = model
        self._loader = loader
        self._index_configs = self._build_index_configs(embeddable_labels)

    @staticmethod
    def _build_index_configs(
        embeddable_labels: list[str] | None,
    ) -> tuple[tuple[str, str], ...]:
        """Build index configs from label list, falling back to defaults."""
        if embeddable_labels is None:
            return INDEX_CONFIGS
        return tuple(
            (f"{label.lower()}_embedding_index", label)
            for label in embeddable_labels
        )

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
        """Embed items and store embeddings on Neo4j nodes.

        Matching always uses the label's full composite key from
        ``EMBEDDING_KEY_MAP``, not ``match_prop`` alone — a single property
        (e.g. ``code``) is ambiguous for composite-identity nodes like
        ``:Term {vocabulary_name, code}``. ``match_prop`` is retained for
        backward-compatible call sites but no longer drives matching.
        """
        if not items:
            return

        texts = [text_fn(item) for item in items]
        embeddings = self.embed_batch(texts)
        match_keys = EMBEDDING_KEY_MAP.get(label, (match_prop,))

        assert self._loader is not None
        for item, embedding in zip(items, embeddings):
            match = _build_match(label, match_keys, item)
            if match is None:
                continue
            self._loader.set_node_embedding(
                label=label, match=match, embedding=list(embedding),
            )

    def create_all_indexes(self, dimensions: int = 1536) -> None:
        """Create vector indexes for all embeddable node types."""
        assert self._loader is not None
        for index_name, label in self._index_configs:
            self._loader.create_vector_index(
                index_name, label, dimensions,
            )
