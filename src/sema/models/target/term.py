"""TargetTermDecl — adapter-supplied controlled-vocabulary term."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sema.models.target.refs import ExternalRef, VocabularyRef


class TargetTermDecl(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    vocabulary: VocabularyRef
    code: str = Field(min_length=1)
    display: str = Field(min_length=1)
    domain: str | None = None
    # Unset concept fields stay out of dumps so pre-concept snapshot hashes hold.
    definition: str | None = Field(default=None, exclude_if=lambda v: v is None)
    synonyms: list[str] = Field(default_factory=list, exclude_if=lambda v: not v)
    external_refs: list[ExternalRef] = Field(
        default_factory=list, exclude_if=lambda v: not v
    )
