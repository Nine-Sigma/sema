"""WriteOp dataclasses recorded by graph writers.

Pydantic-frozen data shapes returned and inspected by tests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EntityOp(_Op):
    op: Literal["entity"] = "entity"
    target_model_id: str
    target_model_version: str
    target_schema_snapshot_hash: str
    qualified_name: str
    kind: str
    enrichment_status: dict[str, str]
    is_current: bool = True


class PropertyOp(_Op):
    op: Literal["property"] = "property"
    target_model_id: str
    target_model_version: str
    target_schema_snapshot_hash: str
    parent_entity_qualified_name: str
    name: str
    type: str
    nullable: bool
    synonyms: list[str] = Field(default_factory=list)
    decoded_values: dict[str, str] = Field(default_factory=dict)
    property_kind: str = "COLUMN"
    endpoint_role: str | None = None
    endpoint_target_entity_qualified_name: str | None = None
    endpoint_cardinality: str | None = None
    endpoint_nullable: bool | None = None
    materialized_as_edge_property: bool = True
    is_current: bool = True


class TermOp(_Op):
    op: Literal["term"] = "term"
    target_model_id: str
    target_model_version: str
    target_schema_snapshot_hash: str
    vocabulary_name: str
    code: str
    display: str


class ConstraintOp(_Op):
    op: Literal["constraint"] = "constraint"
    target_model_id: str
    target_model_version: str
    target_schema_snapshot_hash: str
    attached_property_id: str
    constraint_kind: str
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str
    is_current: bool = True


class TargetObligationOp(_Op):
    op: Literal["target_obligation"] = "target_obligation"
    target_model_id: str
    target_model_version: str
    target_schema_snapshot_hash: str
    target_entity: str
    payload: dict[str, Any]


class EnrichmentDecisionOp(_Op):
    op: Literal["enrichment_decision"] = "enrichment_decision"
    target_model_id: str
    target_model_version: str
    target_schema_snapshot_hash: str
    entity_ref: str
    decisions_json: str
    decided_at: datetime
    is_current: bool = True


class RelationshipOp(_Op):
    op: Literal["relationship"] = "relationship"
    rel_type: str
    target_schema_snapshot_hash: str
    from_label: str
    from_keys: dict[str, str]
    to_label: str
    to_keys: dict[str, str]


class VocabularyBindingOp(_Op):
    op: Literal["vocabulary_binding"] = "vocabulary_binding"
    target_model_id: str
    target_model_version: str
    target_schema_snapshot_hash: str
    parent_entity_qualified_name: str
    property_name: str
    vocabulary_name: str
    vocabulary_source: str
    domain: str | None = None
    require_standard: bool = False
    allow_zero_default: bool = False
    standard_domain_governed: bool = False
    effective_date_ref: str | None = None
    resolver_policy_ref: str | None = None
    is_current: bool = True


class ContextCardOp(_Op):
    op: Literal["context_card"] = "context_card"
    target_model_id: str
    target_model_version: str
    target_schema_snapshot_hash: str
    entity_qualified_name: str
    card_version: str
    card_hash: str
    description: str
    examples: list[str] = Field(default_factory=list)
    obligation_summary: str | None = None
    curated_synonyms: list[str] = Field(default_factory=list)
    is_current: bool = True


class CurrentFlipOp(_Op):
    """Records logical-artifact identity tuples whose prior generations
    should have `is_current=false` after this load. Scoped strictly to
    artifacts touched by the current load (lazy-load preservation)."""

    op: Literal["current_flip"] = "current_flip"
    target_model_id: str
    target_model_version: str
    current_snapshot_hash: str
    entity_qualified_names: tuple[str, ...]
    property_keys: tuple[tuple[str, str], ...] = Field(default_factory=tuple)
    obligation_target_entities: tuple[str, ...] = Field(default_factory=tuple)
    enrichment_entity_refs: tuple[str, ...] = Field(default_factory=tuple)
    vocabulary_binding_keys: tuple[tuple[str, str, str], ...] = Field(
        default_factory=tuple
    )
    context_card_keys: tuple[tuple[str, str], ...] = Field(default_factory=tuple)
    term_keys: tuple[tuple[str, str], ...] = Field(default_factory=tuple)


WriteOp = (
    EntityOp
    | PropertyOp
    | TermOp
    | ConstraintOp
    | TargetObligationOp
    | EnrichmentDecisionOp
    | VocabularyBindingOp
    | ContextCardOp
    | RelationshipOp
    | CurrentFlipOp
)
