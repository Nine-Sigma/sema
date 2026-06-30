"""Composite identity for :Term nodes.

Term nodes are keyed on ``{vocabulary_name, code}`` so the same code in
two vocabularies (e.g. ``M`` = Male vs ``M`` = Mississippi) never
collapses into one node. Terms with no known vocabulary use the
``UNSCOPED_VOCAB`` sentinel so the namespace component of the key is
never null.

Pure stdlib — no sema imports — so the models/graph layers can import it
without risking an import cycle.
"""

from __future__ import annotations

UNSCOPED_VOCAB = "_unscoped"


def term_namespace(vocabulary_name: str | None) -> str:
    """Resolve a Term's vocabulary namespace; never empty or null."""
    return vocabulary_name or UNSCOPED_VOCAB
