"""Value-level cross-layer bridge (S2-06 / Finding 1).

Writes the ``(source :Term)-[:MAPS_TO_CONCEPT]->(concept :Term)`` edge — the
actual join for "ask in source terms" — from the value-mapping store. Produced
by neither the source loader nor the target loader today. Both endpoints use
canonical ``{vocabulary_name, code}`` identity, so the edge attaches to the
same term nodes the source build and the concept value-set loader wrote (never
a duplicate).

Generic: source/concept vocabularies and codes are passed as data.
"""

from __future__ import annotations

from sema.graph.loader import GraphLoader
from sema.graph.term_identity_utils import term_namespace


def write_value_bridge(
    loader: GraphLoader,
    *,
    source_vocabulary: str | None,
    source_code: str,
    concept_vocabulary: str,
    concept_code: str,
    source_schema: str,
) -> None:
    """MERGE both canonical Terms and the scoped ``MAPS_TO_CONCEPT`` edge."""
    loader._run(
        "MERGE (src:Term {vocabulary_name: $sv, code: $sc}) "
        "MERGE (concept:Term {vocabulary_name: $cv, code: $cc}) "
        "MERGE (src)-[:MAPS_TO_CONCEPT {source_schema: $ss}]->(concept)",
        sv=term_namespace(source_vocabulary),
        sc=source_code,
        cv=term_namespace(concept_vocabulary),
        cc=concept_code,
        ss=source_schema,
    )


__all__ = ["write_value_bridge"]
