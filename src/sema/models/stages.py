"""Typed intermediate results for the staged L2 pipeline (A → B → C → merge).

These are internal intermediates — NOT assertion-producing outputs.
Assertions are materialized once at the merge step.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StageAResult(BaseModel):
    """Stage A output: table-level entity and grain hypothesis.

    Internal state consumed by Stage B and the merge step.
    Does NOT produce assertions directly.
    """

    primary_entity: str
    grain_hypothesis: str
    synonyms: list[str] = Field(default_factory=list)
    secondary_entity_hints: list[str] = Field(default_factory=list)
    ambiguity_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class StageBColumnResult(BaseModel):
    """Stage B output for a single column.

    Internal state consumed by Stage C triggers and the merge step.
    """

    column: str
    canonical_property_label: str
    semantic_type: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.75)
    synonyms: list[str] = Field(default_factory=list)
    candidate_vocab_families: list[str] = Field(default_factory=list)
    entity_role: str | None = None
    grain_confirmation: str | None = None
    needs_stage_c: bool = False
    ambiguity_notes: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class StageBBatchResult(BaseModel):
    """Stage B output for a column batch (LLM response schema)."""

    columns: list[StageBColumnResult]
    grain_correction: str | None = None
    entity_correction: str | None = None


class UnresolvedColumn(BaseModel):
    """A column that Stage B could not classify."""

    column: str
    reason: Literal["execution_failure", "semantic_unresolved"]
    tier: Literal["critical", "important", "peripheral"]


class StageBCoverage(BaseModel):
    """Coverage metrics for Stage B classification."""

    classified: int
    total: int
    pct: float = Field(ge=0.0, le=1.0)


class StageBResult(BaseModel):
    """Aggregated Stage B result across all batches."""

    status: Literal["B_SUCCESS", "B_PARTIAL", "B_FAILED"]
    batch_results: list[StageBBatchResult] = Field(default_factory=list)
    raw_coverage: StageBCoverage
    critical_coverage: StageBCoverage
    unresolved_columns: list[UnresolvedColumn] = Field(
        default_factory=list,
    )
    retries_used: int = 0
    splits_used: int = 0
    rescues_used: int = 0


class StageCResult(BaseModel):
    """Stage C output: value interpretation for a single column.

    Internal state consumed by the merge step.
    """

    column: str
    decoded_categories: list[dict[str, str]] = Field(
        default_factory=list,
    )
    uncertainty: float = Field(ge=0.0, le=1.0, default=0.0)
    codebook_lookup_needed: bool = False
    normalized_meanings: list[str] = Field(default_factory=list)


class StageCBatchResult(BaseModel):
    """Stage C batch output wrapping multiple column results."""

    columns: list[StageCResult] = Field(default_factory=list)


class StageStatus(BaseModel):
    """Per-table metadata tracking stage outcomes and recovery effort."""

    stage_a: Literal["success", "failed"]
    stage_b_status: Literal["success", "partial", "failed"]
    stage_b_raw_coverage: StageBCoverage
    stage_b_critical_coverage: StageBCoverage
    stage_b_unresolved_columns: list[UnresolvedColumn] = Field(
        default_factory=list,
    )
    stage_b_retries_used: int = 0
    stage_b_splits_used: int = 0
    stage_b_rescues_used: int = 0
    stage_c_triggered: bool = False
    stage_c_columns_requested: int = 0
    stage_c_columns_succeeded: int = 0
    partial_output: bool = False
    warnings: list[str] = Field(default_factory=list)
