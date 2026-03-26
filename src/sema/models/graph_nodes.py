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


class Catalog(BaseModel):
    name: str


class Schema(BaseModel):
    name: str
    catalog: str


class Table(BaseModel):
    name: str
    schema_name: str
    catalog: str
    table_type: str = "TABLE"
    comment: str | None = None


class Column(BaseModel):
    name: str
    table_name: str
    data_type: str
    nullable: bool = True
    comment: str | None = None


class Entity(BaseModel):
    name: str
    description: str | None = None
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolved_at: datetime | None = None


class Property(BaseModel):
    name: str
    description: str | None = None
    semantic_type: SemanticType
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolved_at: datetime | None = None


class Metric(BaseModel):
    name: str
    description: str | None = None
    formula: str | None = None
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolved_at: datetime | None = None


class Term(BaseModel):
    code: str
    label: str
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolved_at: datetime | None = None


class ValueSet(BaseModel):
    name: str


class Synonym(BaseModel):
    text: str
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolved_at: datetime | None = None


class Transformation(BaseModel):
    name: str
    transform_type: str
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolved_at: datetime | None = None
