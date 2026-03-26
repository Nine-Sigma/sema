from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    assertion_ids: list[str] = Field(default_factory=list)


class ResolvedProperty(BaseModel):
    name: str
    semantic_type: str
    physical_column: str
    physical_table: str
    description: str | None = None
    provenance: Provenance


class ResolvedEntity(BaseModel):
    name: str
    description: str | None = None
    physical_table: str
    properties: list[ResolvedProperty] = Field(default_factory=list)
    provenance: Provenance


class PhysicalAsset(BaseModel):
    model_config = {"protected_namespaces": ()}

    catalog: str
    schema: str  # type: ignore[assignment]
    table: str
    columns: list[str] = Field(default_factory=list)


class JoinPredicate(BaseModel):
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    operator: str = "="


class JoinPath(BaseModel):
    from_table: str
    to_table: str
    join_predicates: list[JoinPredicate] = Field(default_factory=list)
    hop_count: int = 1
    cardinality_hint: str | None = None
    sql_snippet: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class GovernedValue(BaseModel):
    property_name: str
    column: str
    table: str
    values: list[dict[str, str]] = Field(default_factory=list)


class ResolvedMetric(BaseModel):
    name: str
    description: str | None = None
    formula: str | None = None
    provenance: Provenance


class ResolvedTransformation(BaseModel):
    name: str
    transform_type: str
    depends_on: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    provenance: Provenance


class AncestryTerm(BaseModel):
    code: str
    label: str
    parent_code: str | None = None


class SemanticCandidateSet(BaseModel):
    query: str
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class SemanticContextObject(BaseModel):
    entities: list[ResolvedEntity] = Field(default_factory=list)
    metrics: list[ResolvedMetric] = Field(default_factory=list)
    transformations: list[ResolvedTransformation] = Field(default_factory=list)
    physical_assets: list[PhysicalAsset] = Field(default_factory=list)
    join_paths: list[JoinPath] = Field(default_factory=list)
    governed_values: list[GovernedValue] = Field(default_factory=list)
    ancestry: list[AncestryTerm] = Field(default_factory=list)
    consumer_hint: str = "nl2sql"
    retrieval_rationale: str | None = None
