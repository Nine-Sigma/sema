"""Target-side typed references."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from sema.models.planner._enums import TargetArtifactKind

QualifiedName = Annotated[
    str,
    StringConstraints(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+$"),
]


class TargetEntityRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_model_id: str = Field(min_length=1)
    qualified_name: QualifiedName
    kind: TargetArtifactKind


class TargetPropertyRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_ref: TargetEntityRef
    property_name: str = Field(min_length=1)


class VocabularySource(str, Enum):
    INLINE = "INLINE"
    EXTERNAL = "EXTERNAL"


class VocabularyRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    source: VocabularySource
