"""TargetTermDecl — adapter-supplied controlled-vocabulary term."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sema.models.target.refs import VocabularyRef


class TargetTermDecl(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    vocabulary: VocabularyRef
    code: str = Field(min_length=1)
    display: str = Field(min_length=1)
    domain: str | None = None
