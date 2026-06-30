"""OMOP/OncoTree resolver policy — the ONLY module where these literals live.

R29 allowlists ``src/sema/resolve/policies/`` precisely so the source
vocabulary ("OncoTree"), the standardizing relationship ("Maps to"), and the
standard flag ("S") are confined here. The generic spine
(:mod:`sema.resolve.policy`) never names them.
"""

from __future__ import annotations

from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.resolve.policy import ResolverPolicy
from sema.resolve.vocab_store_utils import VocabStoreSchema

OMOP_ONCOTREE_CONDITION_REF = "omop.oncotree_to_snomed_condition"

# The OMOP physical schema for the concept-vocabulary tables. This is the only
# place these OMOP column literals live (R29-allowlisted); the VocabStore reads
# them as config so the query layer itself stays vocabulary-agnostic.
OMOP_VOCAB_SCHEMA = VocabStoreSchema(
    concept_table="concept",
    relationship_table="concept_relationship",
    synonym_table="concept_synonym",
    id_col="concept_id",
    code_col="concept_code",
    name_col="concept_name",
    vocab_col="vocabulary_id",
    domain_col="domain_id",
    standard_col="standard_concept",
    invalid_col="invalid_reason",
    rel_from_col="concept_id_1",
    rel_to_col="concept_id_2",
    rel_id_col="relationship_id",
    synonym_name_col="concept_synonym_name",
)


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
