"""TargetPropertyDecl with optional endpoint-property typing fields."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sema.models.target.refs import ConceptRef, ExternalRef

_RESERVED_ENDPOINT_NAMES = frozenset({"subject", "object"})


class PropertyKind(str, Enum):
    COLUMN = "COLUMN"
    ENDPOINT = "ENDPOINT"
    CONCEPT = "CONCEPT"


class TargetPropertyDecl(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    nullable: bool
    synonyms: list[str] = Field(default_factory=list)
    decoded_values: dict[str, str] = Field(default_factory=dict)
    vocabulary_binding: str | None = None
    property_kind: PropertyKind = PropertyKind.COLUMN
    endpoint_role: Literal["subject", "object"] | None = None
    endpoint_target_entity_qualified_name: str | None = None
    endpoint_cardinality: Literal["one", "many"] | None = None
    endpoint_nullable: bool | None = None
    materialized_as_edge_property: bool = True
    # A consumer policy; a denied column is still declared and stored in the graph.
    exposure: Literal["denied"] | None = Field(default=None, exclude_if=lambda v: v is None)
    # Unset concept fields stay out of dumps so pre-concept snapshot hashes hold.
    concept_ref: ConceptRef | None = Field(default=None, exclude_if=lambda v: v is None)
    definition: str | None = Field(default=None, exclude_if=lambda v: v is None)
    external_refs: list[ExternalRef] = Field(
        default_factory=list, exclude_if=lambda v: not v
    )

    @model_validator(mode="after")
    def _validate_kind_consistency(self) -> Self:
        is_endpoint = self.property_kind is PropertyKind.ENDPOINT
        if not is_endpoint and self.name in _RESERVED_ENDPOINT_NAMES:
            raise ValueError(
                f"property name {self.name!r} is reserved for synthesized endpoint "
                f"properties; columnar adapters MUST NOT declare it"
            )
        if is_endpoint:
            missing = [
                f
                for f, v in (
                    ("endpoint_role", self.endpoint_role),
                    ("endpoint_target_entity_qualified_name", self.endpoint_target_entity_qualified_name),
                    ("endpoint_cardinality", self.endpoint_cardinality),
                    ("endpoint_nullable", self.endpoint_nullable),
                )
                if v is None
            ]
            if missing:
                raise ValueError(
                    f"ENDPOINT property requires endpoint fields: missing {missing}"
                )
        return self


