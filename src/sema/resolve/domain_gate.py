"""§4 step 3 — the deterministic domain gate (constraint layer).

Rejects any standardized candidate whose domain differs from the binding's
target domain. A pure in-memory filter over rows already fetched in step 2 — the
domain is read off the concept row, so no extra query is needed.
"""

from __future__ import annotations

from sema.resolve.policy import Candidate, ResolverPolicy
from sema.resolve.vocab_store_utils import ConceptRow


def apply_domain_gate(
    policy: ResolverPolicy, concepts: list[ConceptRow]
) -> list[ConceptRow]:
    """Keep only concepts whose domain matches the target obligation."""
    return [c for c in concepts if policy.in_target_domain(_as_candidate(c))]


def _as_candidate(row: ConceptRow) -> Candidate:
    return Candidate(
        standard_value=row.standard,
        is_invalid=row.invalid_reason is not None,
        domain=row.domain,
    )
