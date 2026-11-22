"""Unenforced entity-level joins and domains, apart from obligation constraints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DeclarationEvidence = Literal["declared", "model"]


class DeclaredJoinDecl(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    referenced_entity: str = Field(min_length=1)
    join_keys: list[tuple[str, str]] = Field(min_length=1)
    evidence: DeclarationEvidence


class DeclaredDomainDecl(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    property_name: str = Field(min_length=1)
    domain_id: str = Field(min_length=1)
    evidence: DeclarationEvidence
