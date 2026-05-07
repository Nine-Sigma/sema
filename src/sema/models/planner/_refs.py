"""Typed references used by planner payloads.

A reference is a structured pointer into the source/target model graph;
producers MUST construct these rather than passing bare strings.

`RefStr` is the lightweight string form: a dot-delimited path
(``scope.entity.field`` or ``scope.field``) that the constraint layer can
resolve back to a node in the source/target model graph. Truly bare strings
("x", "1") are rejected at construction. Callers that already hold a typed
``PropertyRef`` / ``EntityRef`` / ``TermRef`` MAY pass it directly to the
fields that accept ``RefStr | <typed>`` unions.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

# Dot-delimited structured path. At least one dot is required; segments
# allow [_+\-=] for vocab-domain descriptors like
# "omop.condition_occurrence.condition_concept_id.domain=Condition".
_REF_PATTERN = r"^[A-Za-z][A-Za-z0-9_+\-]*(\.[A-Za-z0-9_+\-=]+)+$"

RefStr = Annotated[
    str,
    StringConstraints(min_length=1, pattern=_REF_PATTERN),
]


class SourceRef(BaseModel):
    source_id: str = Field(min_length=1)


class TargetRef(BaseModel):
    target_model_id: str = Field(min_length=1)


class EntityRef(BaseModel):
    model_role: str = Field(pattern=r"^(SOURCE|TARGET)$")
    scope_id: str = Field(min_length=1)
    entity_name: str = Field(min_length=1)


class PropertyRef(BaseModel):
    model_role: str = Field(pattern=r"^(SOURCE|TARGET)$")
    scope_id: str = Field(min_length=1)
    entity_name: str = Field(min_length=1)
    property_name: str = Field(min_length=1)


class TermRef(BaseModel):
    model_role: str = Field(pattern=r"^(SOURCE|TARGET)$")
    scope_id: str = Field(min_length=1)
    code: str = Field(min_length=1)
