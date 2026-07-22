"""§4 step 1a — candidate generation (the code-bearing short-circuit).

A code-bearing source value is matched by exact concept code WITHIN THE SOURCE
vocabulary named by the policy (R9) — never the binding's target vocabulary.
The hop into the target vocabulary happens later, in standardization. No
embeddings, no vector recall: that free-text recall arm is out of scope for the
closed-set Slice-0 path.
"""

from __future__ import annotations

from sema.resolve.policy import ResolverPolicy
from sema.resolve.vocab_store import VocabStore
from sema.resolve.vocab_store_utils import ConceptRow


def generate_candidates(
    store: VocabStore, policy: ResolverPolicy, source_code: str
) -> list[ConceptRow]:
    """Exact source-code match; an unknown code yields no candidates."""
    vocabulary, code = policy.candidate_lookup(source_code)
    row = store.concept_by_code(vocabulary, code)
    return [row] if row is not None else []
