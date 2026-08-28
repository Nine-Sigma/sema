"""`ConceptSource` port (G-concept) — domain-neutral concept enrichment.

The value-mapping store holds concept *codes* only; their human-readable name,
synonyms, and hierarchical ancestors must be fetched to make a concept
retrievable (lexical/vector search, ancestor context). Core depends only on
this port; a showcase adapter implements it against a concrete vocabulary
store. The signature names no target vocabulary — a "concept" here is just a
code in some vocabulary with a name, synonyms, and broader codes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ConceptDetail:
    code: str
    name: str
    synonyms: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ConceptAncestor:
    code: str
    name: str


class ConceptSource(Protocol):
    """Fetches names/synonyms and ancestors for concept codes."""

    def name_synonyms(self, code: str) -> ConceptDetail | None: ...

    def ancestors(self, code: str) -> list[ConceptAncestor]: ...


__all__ = ["ConceptAncestor", "ConceptDetail", "ConceptSource"]
