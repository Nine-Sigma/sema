"""TargetModelDescriptor — adapter-supplied identity + completeness."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from sema.models.target.completeness import SemanticCompletenessAnnotations

KebabCaseId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9-]*$"),
]


class TargetModelDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_model_id: KebabCaseId
    target_model_version: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    owner: str | None = None
    vocabulary_release: str | None = None
    completeness: SemanticCompletenessAnnotations
