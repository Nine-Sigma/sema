"""§4 step 2 — 'Maps to' standardization (OMOP-correct standard selection).

Each source candidate is walked to its standard target concept(s) via the
policy's standardizing relationship, keeping only valid concepts. The
relationship name and standard flag are policy data, never hardcoded here. A
candidate that is already standard-and-valid is itself a target.

Validity-window filtering reduces to ``invalid_reason IS NULL`` because the
store's :class:`ConceptRow` carries no validity dates; date-window gating is
added when the row shape grows date columns (Slice-0 inputs carry no event
date, so the window check is a no-op today).
"""

from __future__ import annotations

from sema.resolve.engine_utils import dedupe_concepts
from sema.resolve.policy import Candidate, ResolverPolicy
from sema.resolve.vocab_store import VocabStore
from sema.resolve.vocab_store_utils import ConceptRow


def standardize(
    store: VocabStore, policy: ResolverPolicy, candidate: ConceptRow
) -> list[ConceptRow]:
    """Return the standard, valid target concepts for one source candidate."""
    standard_flag = policy.standard_flag if policy.require_standard else None
    targets = store.maps_to_targets(
        candidate.id,
        relationship_id=policy.maps_to_relationship,
        standard_flag=standard_flag,
        only_valid=True,
    )
    if _is_standard_target(policy, candidate):
        return dedupe_concepts([candidate, *targets])
    return dedupe_concepts(targets)


def _is_standard_target(policy: ResolverPolicy, candidate: ConceptRow) -> bool:
    probe = Candidate(
        standard_value=candidate.standard,
        is_invalid=candidate.invalid_reason is not None,
        domain=candidate.domain,
    )
    return policy.is_standard(probe) and policy.is_valid(probe)
