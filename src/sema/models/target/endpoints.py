"""GRAPH_EDGE endpoint declarations.

Adapters declare `endpoints.subject` and `endpoints.object` on
`GRAPH_EDGE` entities. The normalizer compiles these into reserved
endpoint `Property` instances; adapters never construct endpoint
properties directly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from sema.models.target.refs import TargetEntityRef


class EdgeEndpointDecl(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["subject", "object"]
    target_entity: TargetEntityRef
    cardinality: Literal["one", "many"] = "one"
    nullable: bool = False


class EdgeEndpointsDecl(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: EdgeEndpointDecl
    object: EdgeEndpointDecl
