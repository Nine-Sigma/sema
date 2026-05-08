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
    effective_date_ref: str | None = None
    resolver_policy_ref: str | None = None
