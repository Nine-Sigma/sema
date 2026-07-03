"""Shared OMOP condition VocabularyBindingDecl builder for live tests.

Lives under ``tests/`` (R29-allowlisted), so the OMOP-specific literals here
are legitimate test scaffolding, not engine-core leakage.
"""

from __future__ import annotations

from sema.models.planner._enums import TargetArtifactKind
from sema.models.target.refs import TargetEntityRef, VocabularyRef, VocabularySource
from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.resolve.policies.omop import OMOP_ONCOTREE_CONDITION_REF


def build_condition_binding() -> VocabularyBindingDecl:
    return VocabularyBindingDecl(
        entity_ref=TargetEntityRef(
            target_model_id="omop_condition_slice0",
            qualified_name="omop.condition_occurrence",
            kind=TargetArtifactKind.TABLE_ROW,
        ),
        property_name="condition_concept_id",
        vocabulary=VocabularyRef(
            name="OMOP-Condition", source=VocabularySource.EXTERNAL
        ),
        domain="Condition",
        require_standard=True,
        allow_zero_default=False,
        resolver_policy_ref=OMOP_ONCOTREE_CONDITION_REF,
    )
