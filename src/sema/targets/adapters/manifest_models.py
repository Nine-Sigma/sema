"""Pydantic models mirroring the manifest schema (manifest_version=1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sema.models.planner._enums import (
    PrimaryKeyStrategy,
    TargetArtifactKind,
)
from sema.models.target.completeness import SemanticCompletenessAnnotations
from sema.models.target.refs import VocabularySource


class ManifestVocabulary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    source: VocabularySource


class ManifestTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vocabulary: str = Field(min_length=1)
    code: str = Field(min_length=1)
    display: str = Field(min_length=1)
    domain: str | None = None


class ManifestVocabularyBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vocabulary: str = Field(min_length=1)
    domain: str | None = None
    require_standard: bool = False
    allow_zero_default: bool = False
    effective_date_ref: str | None = None
    resolver_policy_ref: str | None = None


class ManifestProperty(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    nullable: bool = False
    synonyms: list[str] = Field(default_factory=list)
    decoded_values: dict[str, str] = Field(default_factory=dict)
    vocabulary_binding: ManifestVocabularyBinding | None = None


class ManifestEndpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_entity: str = Field(min_length=1)
    cardinality: Literal["one", "many"] = "one"
    nullable: bool = False


class ManifestEndpoints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: ManifestEndpoint
    object: ManifestEndpoint


class ManifestForeignKey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    referenced_entity: str = Field(min_length=1)
    join_keys: list[tuple[str, str]] = Field(min_length=1)
    same_build_required: bool = True


class ManifestDomainConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_name: str = Field(min_length=1)
    domain_id: str = Field(min_length=1)


class ManifestRowClause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["presence", "equality"]
    field: str = Field(min_length=1)
    value: Any | None = None


class ManifestRowPredicate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["AND", "OR"]
    clauses: list[ManifestRowClause] = Field(min_length=1)


class ManifestExternalSequence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mapping_table_name: str = Field(min_length=1)
    canonical_identity_column: str = Field(min_length=1)
    sequence_column: str = Field(min_length=1)


class ManifestObligation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_fields: list[str] = Field(min_length=1)
    nullable_fields: list[str] = Field(default_factory=list)
    primary_key: PrimaryKeyStrategy
    external_sequence: ManifestExternalSequence | None = None
    foreign_keys: list[ManifestForeignKey] = Field(default_factory=list)
    domain_constraints: list[ManifestDomainConstraint] = Field(default_factory=list)
    allowed_defaults: dict[str, Any] = Field(default_factory=dict)
    minimum_viable_row: ManifestRowPredicate | None = None


class ManifestContextCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card_version: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=4000)
    examples: list[str] = Field(default_factory=list)
    obligation_summary: str | None = None
    curated_synonyms: list[str] = Field(default_factory=list)


class ManifestEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qualified_name: str = Field(min_length=1)
    kind: TargetArtifactKind
    completeness: SemanticCompletenessAnnotations | None = None
    endpoints: ManifestEndpoints | None = None
    properties: list[ManifestProperty] = Field(default_factory=list)
    obligation: ManifestObligation | None = None
    context_card: ManifestContextCard | None = None


class ManifestDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_model_id: str = Field(min_length=1)
    target_model_version: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    owner: str | None = None
    vocabulary_release: str | None = None
    completeness: SemanticCompletenessAnnotations | None = None


class ParsedManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: int
    descriptor: ManifestDescriptor
    vocabularies: list[ManifestVocabulary] = Field(default_factory=list)
    terms: list[ManifestTerm] = Field(default_factory=list)
    entities: list[ManifestEntity] = Field(default_factory=list)
