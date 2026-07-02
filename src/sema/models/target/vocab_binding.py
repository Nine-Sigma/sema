"""VocabularyBindingDecl — adapter-declared property→vocabulary binding."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sema.models.target.refs import TargetEntityRef, VocabularyRef


class VocabularyBindingDecl(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_ref: TargetEntityRef
    property_name: str = Field(min_length=1)
    vocabulary: VocabularyRef
    domain: str | None = None
    require_standard: bool = False
    allow_zero_default: bool = False
    standard_domain_governed: bool = Field(
        default=False,
        description=(
            "When true, the `vocabulary` field is a legacy/governance ANCHOR "
            "only: resolver acceptance is governed by standardness + target "
            "domain, NOT by the target vocabulary id. A source code may resolve "
            "to a standard concept in the target domain via any vocabulary "
            "(e.g. an OncoTree code -> a standard Condition concept in SNOMED "
            "OR ICDO3), and both are correct."
        ),
    )
    effective_date_ref: str | None = None
    resolver_policy_ref: str | None = None
