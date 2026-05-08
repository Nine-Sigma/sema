"""LoadedTarget — value object returned by load_target()."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from sema.models.target.context_card import LoadedContextCard
from sema.models.target.descriptor import TargetModelDescriptor
from sema.models.target.enrichment import EnrichmentDecisionRecord
from sema.models.target.refs import TargetEntityRef


class LoadedTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    descriptor: TargetModelDescriptor
    target_schema_snapshot_hash: str = Field(min_length=64, max_length=64)
    entity_refs: list[TargetEntityRef] = Field(default_factory=list)
    enrichment_decisions: list[EnrichmentDecisionRecord] = Field(default_factory=list)
    card_versions: dict[str, str] = Field(default_factory=dict)
    aggregate_context_card_version: str = Field(min_length=1)
    context_cards: list[LoadedContextCard] = Field(default_factory=list)
    card_hashes: dict[str, str] = Field(default_factory=dict)
    materialized_at: datetime
