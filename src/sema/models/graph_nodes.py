from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SemanticType(str, Enum):
    IDENTIFIER = "identifier"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    NUMERIC = "numeric"
    FREE_TEXT = "free_text"


class DataSource(BaseModel):
    id: str
    ref: str
    platform: str
    workspace: str


class Catalog(BaseModel):
    id: str
    ref: str
    name: str


class Schema(BaseModel):
    id: str
    ref: str
    name: str
    catalog: str


class Table(BaseModel):
    id: str
    ref: str
    name: str
    schema_name: str
    catalog: str
    table_type: str = "TABLE"
    comment: str | None = None


class Column(BaseModel):
    id: str
    ref: str
    name: str
    table_name: str
    data_type: str
    nullable: bool = True
    comment: str | None = None


class Entity(BaseModel):
    id: str
    name: str
    description: str | None = None
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolved_at: datetime | None = None
    embedding_updated_at: datetime | None = None


class Property(BaseModel):
    id: str
    name: str
    description: str | None = None
    semantic_type: SemanticType
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolved_at: datetime | None = None
    embedding_updated_at: datetime | None = None


class Metric(BaseModel):
    id: str
    name: str
    description: str | None = None
    formula: str | None = None
    grain: str | None = None
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolved_at: datetime | None = None
    embedding_updated_at: datetime | None = None


class Term(BaseModel):
    id: str
    code: str
    label: str
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolved_at: datetime | None = None
    embedding_updated_at: datetime | None = None


class ValueSet(BaseModel):
    id: str
    name: str


class Alias(BaseModel):
    id: str
    text: str
    description: str | None = None
    is_preferred: bool
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolved_at: datetime | None = None
    embedding_updated_at: datetime | None = None


class JoinPredicate(BaseModel):
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    operator: str = "="


class JoinPath(BaseModel):
    id: str
    name: str
    sql_snippet: str | None = None
    join_predicates: list[JoinPredicate]
    hop_count: int
    cardinality_hint: str | None = None
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolved_at: datetime | None = None


class Transformation(BaseModel):
    id: str
    ref: str | None = None
    name: str
    transform_type: str
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolved_at: datetime | None = None
