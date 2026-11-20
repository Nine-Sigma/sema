"""TargetEntityDecl — entity-shape declaration with optional edge endpoints and identity."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sema.models.planner._enums import TargetArtifactKind
from sema.models.target.completeness import SemanticCompletenessAnnotations
from sema.models.target.declared import DeclaredDomainDecl, DeclaredJoinDecl
from sema.models.target.endpoints import EdgeEndpointsDecl
from sema.models.target.identity import EntityIdentity
from sema.models.target.properties import TargetPropertyDecl
from sema.models.target.refs import TargetEntityRef


class TargetEntityDecl(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: TargetEntityRef
    properties: list[TargetPropertyDecl] = Field(default_factory=list)
    completeness: SemanticCompletenessAnnotations | None = None
    endpoints: EdgeEndpointsDecl | None = None
    declared_joins: list[DeclaredJoinDecl] = Field(default_factory=list)
    declared_domains: list[DeclaredDomainDecl] = Field(default_factory=list)
    identity: EntityIdentity | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def _validate_endpoints_kind_invariant(self) -> Self:
        is_edge = self.ref.kind is TargetArtifactKind.GRAPH_EDGE
        if is_edge and self.endpoints is None:
            raise ValueError("GRAPH_EDGE entity requires endpoints")
        if not is_edge and self.endpoints is not None:
            raise ValueError(
                f"endpoints is only valid for GRAPH_EDGE entities; "
                f"kind={self.ref.kind.value} forbids endpoints"
            )
        return self
