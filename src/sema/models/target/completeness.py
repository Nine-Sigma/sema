"""Per-facet semantic-completeness annotations.

A target ontology can be authoritative about some facets and silent about
others; per-facet annotations let the enrichment runner make per-facet
decisions instead of binary all-or-nothing.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class SemanticCompleteness(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    NONE = "NONE"
    EXTERNAL = "EXTERNAL"


class SemanticCompletenessAnnotations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    structure: SemanticCompleteness
    obligations: SemanticCompleteness
    vocabulary_bindings: SemanticCompleteness
    semantic_aliases: SemanticCompleteness
    terms: SemanticCompleteness
