from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from sema.models.target.identity import EntityIdentity


class Provenance(BaseModel):
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    assertion_ids: list[str] = Field(default_factory=list)


class JoinPredicate(BaseModel):
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    operator: str = "="


class LabelSource(BaseModel):
    """Where a coded column's label lives; `via` joins the label's relation when it differs."""

    table: str
    column: str
    via: list[JoinPredicate] = Field(default_factory=list)


class ResolvedProperty(BaseModel):
    name: str
    semantic_type: str
    physical_column: str
    physical_table: str
    description: str | None = None
    vocabulary: str | None = None
    provenance: Provenance
    labels: list[LabelSource] = Field(default_factory=list, exclude_if=lambda v: not v)


class ResolvedEntity(BaseModel):
    name: str
    description: str | None = None
    physical_table: str
    properties: list[ResolvedProperty] = Field(default_factory=list)
    provenance: Provenance
    key_columns: list[str] = Field(default_factory=list, exclude_if=lambda v: not v)
    identity: EntityIdentity | None = Field(default=None, exclude_if=lambda v: v is None)
    null_key_rows: int | None = Field(default=None, exclude_if=lambda v: v is None)


class SharedEntity(BaseModel):
    """An entity owned by another relation that this relation's rows belong to."""

    entity: str
    owner_table: str
    key_columns: list[str]
    identity: EntityIdentity | None = Field(default=None, exclude_if=lambda v: v is None)


class PhysicalAsset(BaseModel):
    model_config = {"protected_namespaces": ()}

    catalog: str
    schema: str  # type: ignore[assignment]
    table: str
    columns: list[str] = Field(default_factory=list)
    key_columns: list[str] = Field(default_factory=list, exclude_if=lambda v: not v)
    shared_entities: list[SharedEntity] = Field(
        default_factory=list, exclude_if=lambda v: not v,
    )


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
    values: list[dict[str, str | None]] = Field(default_factory=list)
    vocabulary: str | None = None
    ambiguous: bool = False


class ResolvedMetric(BaseModel):
    name: str
    description: str | None = None
    formula: str | None = None
    aggregates: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    grains: list[str] = Field(default_factory=list)
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
    vocabulary: str | None = None


class HierarchyAnchor(BaseModel):
    """A matched term whose descendants are members of a column's value set."""

    code: str
    label: str
    vocabulary: str | None = None
    table: str
    column: str
    schema_name: str = ""
    member_count: int
    member_codes: list[str] = Field(default_factory=list)
    score: float = 0.0


class ColumnMatch(BaseModel):
    """A column the question names, and another listed column whose definition states
    that same name: the named column ranks first, the other is its alternative."""

    table: str
    column: str
    alternative_table: str
    alternative_column: str


class ResultShape(BaseModel):
    """The answer's output columns, in order."""

    columns: list[str] = Field(default_factory=list)


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
    hierarchy_anchors: list[HierarchyAnchor] = Field(
        default_factory=list, exclude_if=lambda anchors: not anchors,
    )
    column_matches: list[ColumnMatch] = Field(
        default_factory=list, exclude_if=lambda matches: not matches,
    )
    result_shape: ResultShape | None = Field(
        default=None, exclude_if=lambda shape: shape is None,
    )
    consumer: str = "nl2sql"
    retrieval_rationale: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_consumer_hint(cls, data: Any) -> Any:
        if isinstance(data, dict) and "consumer_hint" in data:
            data.setdefault("consumer", data.pop("consumer_hint"))
        return data
