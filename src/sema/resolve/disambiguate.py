"""§4 step 4 — single-model disambiguation of the ambiguous tail.

The code-bearing hot path never reaches here: an exact source code yields
exactly one standard survivor. Only a genuine >1-survivor tie escalates
(→ Zone-2). Slice 0 ships a single-model disambiguator (no council; the council
is Slice 1+). With no model wired the default declines to pick (returns
``None``) so the tie is surfaced for review rather than silently auto-mapped —
the orchestrator then records a deterministic placeholder under review_pending.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from sema.resolve.vocab_store_utils import ConceptRow


class Disambiguator(Protocol):
    """Picks one survivor from a tie, or ``None`` to escalate for review."""

    def __call__(self, survivors: Sequence[ConceptRow]) -> ConceptRow | None: ...


def pick_single_survivor(survivors: Sequence[ConceptRow]) -> ConceptRow | None:
    """Default: resolve only an unambiguous single survivor; decline ties."""
    return survivors[0] if len(survivors) == 1 else None
