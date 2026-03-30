from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AssertionPredicate(str, Enum):
    TABLE_EXISTS = "table_exists"
    COLUMN_EXISTS = "column_exists"
    HAS_DATATYPE = "has_datatype"
    HAS_LABEL = "has_label"
    HAS_DESCRIPTION = "has_description"
    HAS_COMMENT = "has_comment"
    HAS_TAG = "has_tag"
    HAS_TOP_VALUES = "has_top_values"
    HAS_SAMPLE_ROWS = "has_sample_rows"

    # Deprecated: use HAS_JOIN_EVIDENCE instead
    JOINS_TO = "joins_to"

    HAS_ENTITY_NAME = "has_entity_name"
    HAS_PROPERTY_NAME = "has_property_name"
    HAS_SEMANTIC_TYPE = "has_semantic_type"
    HAS_DECODED_VALUE = "has_decoded_value"

    # Deprecated: use HAS_ALIAS instead
    HAS_SYNONYM = "has_synonym"

    VOCABULARY_MATCH = "vocabulary_match"
    PARENT_OF = "parent_of"
    MAPS_TO = "maps_to"

    HAS_ALIAS = "has_alias"
    HAS_JOIN_EVIDENCE = "has_join_evidence"
    ENTITY_ON_TABLE = "entity_on_table"
    PROPERTY_ON_COLUMN = "property_on_column"


class AssertionStatus(str, Enum):
    AUTO = "auto"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PINNED = "pinned"
    SUPERSEDED = "superseded"


class Assertion(BaseModel):
    id: str
    subject_ref: str
    subject_id: str | None = None
    predicate: AssertionPredicate
    payload: dict[str, Any] = Field(default_factory=dict)
    object_ref: str | None = None
    object_id: str | None = None
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    # DEPRECATED: status is vestigial. The canonical status model uses
    # StatusEvent (see sema.models.lifecycle). This field is kept for
    # backward compatibility with legacy materialization code that reads
    # it directly. New code should use effective_status() from status_store.
    status: AssertionStatus = AssertionStatus.AUTO
    run_id: str
    observed_at: datetime
