"""OMOP/OncoTree resolver policy — the ONLY module where these literals live.

R29 allowlists ``src/sema/resolve/policies/`` precisely so the source
vocabulary ("OncoTree"), the standardizing relationship ("Maps to"), and the
standard flag ("S") are confined here. The generic spine
(:mod:`sema.resolve.policy`) never names them.
"""

from __future__ import annotations

from sema.compile.compiler_utils import StagingColumns
from sema.models.planner._enums import PrimaryKeyStrategy
from sema.models.planner.target_model import TargetObligation
from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.resolve.policy import ResolverPolicy
from sema.resolve.vocab_store_utils import VocabStoreSchema

OMOP_ONCOTREE_CONDITION_REF = "omop.oncotree_condition"

# §1.5(b) staging column names for the OncoTree->OMOP Condition showcase. The
# compiler (R29-scanned) never names these literals; it reads them from here.
OMOP_STAGING_COLUMNS = StagingColumns(
    source_value_column="source_oncotree_code",
    target_concept_column="condition_concept_id",
)

# §1.5(b) staging target. Distinct from the production condition_occurrence
# obligation (US-007): no person_id, no dates, no FK closure. The required
# fields are the staging projection columns that must be value-producing per
# source row (§1.5(e)). This OMOP-named obligation lives in the allowlisted
# policy layer; the generic assembler (R29-scanned) never names these fields.
SLICE0_STAGING_TARGET = "condition_occurrence_staging"
_STAGING_PREFIX = "target.condition_occurrence_staging"

# The three §1.5(e) staging required-field refs. Exposed so the live fit chain
# (pipeline layer) references the SAME refs the assembler's coverage gate checks
# — the target property (a VOCAB_LOOKUP) plus the two run-constant provenance
# columns (CONSTANT field maps). Keeping them here confines the OMOP literals to
# the allowlisted policy layer.
SLICE0_CONDITION_CONCEPT_FIELD = f"{_STAGING_PREFIX}.condition_concept_id"
SLICE0_RESOLVER_POLICY_FIELD = f"{_STAGING_PREFIX}.resolver_policy_ref"
SLICE0_VOCAB_RELEASE_FIELD = f"{_STAGING_PREFIX}.vocab_release"

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
    """Build the OncoTree→OMOP standard-Condition policy bound to ``binding``.

    The target is a valid + standard (``standard_concept='S'``) OMOP concept in
    ``domain_id='Condition'`` — vocabulary-agnostic (SNOMED or ICDO3 both
    acceptable), not SNOMED-specific.

    The target obligation (domain, ``require_standard``, ``allow_zero_default``)
    is carried by ``binding``; only the source-side literals are supplied here.
    """
    return ResolverPolicy(
        source_vocabulary="OncoTree",
        maps_to_relationship="Maps to",
        standard_flag="S",
        binding=binding,
    )


def make_slice0_staging_obligation() -> TargetObligation:
    """Build the Slice-0 staging obligation (§1.5(b)).

    ``condition_concept_id`` stays REQUIRED, not nullable (§1.5(e)): a per-code
    NO_MAP is a per-row store/staging outcome, not an un-covered field.
    """
    return TargetObligation(
        target_entity=SLICE0_STAGING_TARGET,
        required_fields=[
            SLICE0_CONDITION_CONCEPT_FIELD,
            SLICE0_RESOLVER_POLICY_FIELD,
            SLICE0_VOCAB_RELEASE_FIELD,
        ],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
    )
