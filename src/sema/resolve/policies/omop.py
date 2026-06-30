"""OMOP/OncoTree resolver policy — the ONLY module where these literals live.

R29 allowlists ``src/sema/resolve/policies/`` precisely so the source
vocabulary ("OncoTree"), the standardizing relationship ("Maps to"), and the
standard flag ("S") are confined here. The generic spine
(:mod:`sema.resolve.policy`) never names them.
"""

from __future__ import annotations

from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.resolve.policy import ResolverPolicy

OMOP_ONCOTREE_CONDITION_REF = "omop.oncotree_to_snomed_condition"


def make_omop_oncotree_condition_policy(
    binding: VocabularyBindingDecl,
) -> ResolverPolicy:
    """Build the OncoTree→SNOMED Condition policy bound to ``binding``.

    The target obligation (domain, ``require_standard``, ``allow_zero_default``)
    is carried by ``binding``; only the source-side literals are supplied here.
    """
    return ResolverPolicy(
        source_vocabulary="OncoTree",
        maps_to_relationship="Maps to",
        standard_flag="S",
        binding=binding,
    )
