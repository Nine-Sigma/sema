"""Allowlist for embedding match keys.

Embedding writes build Cypher that interpolates a node label and match
property names directly into the query string. To keep that safe and to
fix duplicate-code Term clobbering, the label and every match property
must come from this fixed allowlist — never from node data.

``:Term`` identity is composite ``{vocabulary_name, code}`` (see
``term_identity_utils``), so its embedding must be matched on BOTH
properties; matching on ``code`` alone lets two terms sharing a code
across vocabularies overwrite each other's embedding.

Graph-layer module (no engine imports) so ``loader`` can import it
without a cycle; ``engine.embeddings`` re-exports the map.
"""

from __future__ import annotations

from typing import Final, Iterable

# Label -> ordered tuple of node properties forming its embedding match key.
EMBEDDING_MATCH_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "Entity": ("name",),
    "Property": ("name", "entity_name"),
    "Term": ("vocabulary_name", "code"),
    "Alias": ("text",),
    "Metric": ("name",),
    # Synonym kept for backward compat with existing Neo4j nodes.
    "Synonym": ("text",),
    # Transformation kept for backward compat.
    "Transformation": ("name",),
}

EMBEDDABLE_LABELS: Final[frozenset[str]] = frozenset(EMBEDDING_MATCH_KEYS)

_ALLOWED_PROPS: Final[dict[str, frozenset[str]]] = {
    label: frozenset(props) for label, props in EMBEDDING_MATCH_KEYS.items()
}


def validate_embedding_label(label: str) -> None:
    if label not in EMBEDDABLE_LABELS:
        raise ValueError(
            f"Label {label!r} is not embeddable; "
            f"allowed: {sorted(EMBEDDABLE_LABELS)}"
        )


def validate_match_props(label: str, props: Iterable[str]) -> None:
    validate_embedding_label(label)
    allowed = _ALLOWED_PROPS[label]
    invalid = [p for p in props if p not in allowed]
    if invalid:
        raise ValueError(
            f"Match props {invalid} not allowed for label {label!r}; "
            f"allowed: {sorted(allowed)}"
        )
