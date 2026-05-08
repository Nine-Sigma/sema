"""Enrichment decision shape: facet, status, per-facet decision, per-entity record."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sema.models.target.refs import TargetEntityRef


class Facet(str, Enum):
    structure = "structure"
    obligations = "obligations"
    vocabulary_bindings = "vocabulary_bindings"
    semantic_aliases = "semantic_aliases"
    terms = "terms"


class EnrichmentStatus(str, Enum):
    not_required = "not_required"
    required_deferred = "required_deferred"
    required_skipped = "required_skipped"
    supplied_by_adapter = "supplied_by_adapter"


_ALL_FACETS: frozenset[Facet] = frozenset(Facet)


class FacetDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: EnrichmentStatus
    reason: str = Field(min_length=1)
    decided_at: datetime


class EnrichmentDecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_ref: TargetEntityRef
    decisions: dict[Facet, FacetDecision]
    decided_at: datetime

    @model_validator(mode="after")
    def _validate_all_facets_present(self) -> Self:
        keys = frozenset(self.decisions.keys())
        if keys != _ALL_FACETS:
            missing = sorted(f.value for f in (_ALL_FACETS - keys))
            extra = sorted(f.value for f in (keys - _ALL_FACETS))
            raise ValueError(
                f"EnrichmentDecisionRecord.decisions must cover exactly the five facets; "
                f"missing={missing} extra={extra}"
            )
        return self
