"""Helpers for the staged L2 pipeline (A → B → C → merge).

Prompt builders, coverage computation, critical column identification,
and pass/fail logic for the decomposed semantic interpretation stages.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.stages import (
    StageAResult,
    StageBCoverage,
    StageBColumnResult,
    StageBResult,
    StageCResult,
    UnresolvedColumn,
)

if TYPE_CHECKING:
    from sema.models.domain import DomainContext


@dataclass(frozen=True)
class PromptLayers:
    """Controls which domain-aware prompt layers are active.

    Each flag independently toggles one layer so rollout steps
    2-5 can be evaluated in isolation.
    """

    enable_domain_bias: bool = True
    enable_type_inventory: bool = True
    enable_vocab_hints: bool = True
    enable_few_shot: bool = True
    enable_stage_c: bool = True

_KEY_PATTERN = re.compile(
    r"(?:_id$|_key$|_pk$|_fk$|^id$|^key$)", re.IGNORECASE,
)

_RAW_COVERAGE_THRESHOLD = 0.75


# -- Shared formatting helpers ---------------------------------------------

def _column_sketch(columns: list[dict[str, Any]]) -> str:
    """Format column names and types for Stage A prompt."""
    lines: list[str] = []
    for col in columns:
        line = f"  {col['name']} ({col.get('data_type', 'UNKNOWN')})"
        if col.get("comment"):
            line += f" — {col['comment']}"
        lines.append(line)
    return "\n".join(lines)


def _sample_rows_sketch(
    rows: list[dict[str, Any]], max_rows: int = 5,
) -> str:
    return "\n".join(
        f"  {json.dumps(row)}" for row in rows[:max_rows]
    )


def _column_detail_line(
    col: dict[str, Any], max_values: int = 5,
) -> str:
    """Format a single column with detail for Stage B."""
    line = f"  {col['name']} ({col.get('data_type', 'UNKNOWN')})"
    if col.get("comment"):
        line += f" — {col['comment']}"
    if col.get("top_values"):
        vals = [v["value"] for v in col["top_values"][:max_values]]
        line += f"\n    top values: {', '.join(str(v) for v in vals)}"
    if col.get("null_pct") is not None:
        line += f" | null%: {col['null_pct']}"
    if col.get("distinct_count") is not None:
        line += f" | distinct: {col['distinct_count']}"
    return line


# -- Stage A prompt ---------------------------------------------------------

def build_stage_a_prompt(
    table_metadata: dict[str, Any],
    *,
    domain_context: DomainContext | None = None,
    layers: PromptLayers | None = None,
) -> str:
    """Build the Stage A prompt: entity and grain hypothesis."""
    from sema.engine.domain_prompts import build_domain_bias_header

    _layers = layers or PromptLayers()
    parts: list[str] = []

    # Domain bias header (empty when no domain or layer disabled)
    if _layers.enable_domain_bias:
        header = build_domain_bias_header(domain_context)
        if header:
            parts.append(header)
            parts.append("")

    parts.append(f"Table: {table_metadata['table_name']}")
    if table_metadata.get("comment"):
        parts.append(f"Comment: {table_metadata['comment']}")

    columns = table_metadata.get("columns", [])
    parts.append(f"\nColumns ({len(columns)}):")
    parts.append(_column_sketch(columns))

    sample_rows = table_metadata.get("sample_rows", [])
    if sample_rows:
        parts.append("\nSample rows:")
        parts.append(_sample_rows_sketch(sample_rows))

    parts.append("""
Based on the table name, column names, types, and any sample rows above,
determine what business entity this table represents and what a single
row means (the grain).

Return ONLY valid JSON with:
- "primary_entity": the main entity this table describes
- "grain_hypothesis": what a single row represents (e.g. "one row per patient", "one row per mutation call per sample")
- "synonyms": alternative names someone might search for this entity (e.g. ["tumor sample", "biopsy specimen"])
- "secondary_entity_hints": list of related entities referenced by columns (e.g. ["gene", "protein change"])
- "ambiguity_flags": list of warnings about mixed or unclear granularity (empty if clear)
- "confidence": 0.0–1.0 how confident you are in the entity and grain

Do NOT classify individual columns — that is a separate step.
Do NOT guess vocabularies or ontologies.""")

    # Few-shot examples (domain-specific, empty for zero-shot)
    if _layers.enable_few_shot:
        from sema.engine.few_shot import format_examples
        domain = domain_context.effective_domain if domain_context else None
        fs_block = format_examples(domain=domain, stage="A")
        if fs_block:
            parts.append(f"\n{fs_block}")

    parts.append("\nReturn ONLY valid JSON, no markdown.")

    return "\n".join(parts)


# -- Stage B prompt ---------------------------------------------------------

def build_stage_b_prompt(
    table_metadata: dict[str, Any],
    column_batch: list[dict[str, Any]],
    stage_a: StageAResult,
    *,
    domain_context: DomainContext | None = None,
    layers: PromptLayers | None = None,
) -> str:
    """Build Stage B prompt: property classification for a column batch."""
    from sema.engine.domain_prompts import (
        build_domain_bias_header,
        build_vocab_family_hints,
        get_semantic_type_inventory,
    )

    _layers = layers or PromptLayers()
    parts: list[str] = []

    # Domain bias header
    if _layers.enable_domain_bias:
        header = build_domain_bias_header(domain_context)
        if header:
            parts.append(header)
            parts.append("")

    parts.append(f"Table: {table_metadata['table_name']}")

    parts.append(
        f"\nEntity context from prior analysis:"
        f"\n  Entity: {stage_a.primary_entity}"
        f"\n  Grain: {stage_a.grain_hypothesis}"
    )
    if stage_a.secondary_entity_hints:
        hints = ", ".join(stage_a.secondary_entity_hints)
        parts.append(f"  Secondary entities: {hints}")

    parts.append(f"\nColumns to classify ({len(column_batch)}):")
    for col in column_batch:
        parts.append(_column_detail_line(col))

    if _layers.enable_type_inventory:
        type_inv = get_semantic_type_inventory(domain_context)
    else:
        type_inv = get_semantic_type_inventory(None)
    parts.append(f"""
For each column above, return a JSON object with:
- "columns": array, one per column, each with:
  - "column": exact column name
  - "canonical_property_label": human-readable property name
  - "semantic_type": one of [{type_inv}]
  - "confidence": 0.0–1.0 how confident you are in this classification
  - "synonyms": alternative names for this property (empty list if none)
  - "candidate_vocab_families": list of semantic family labels \
(e.g. "diagnosis coding system", "gene symbol namespace"). \
Do NOT name a specific ontology or coding system unless the column \
header or values explicitly identify it.
  - "entity_role": role in entity \
(e.g. "primary_key", "foreign_key", "attribute", "secondary")
  - "grain_confirmation": confirm or correct the grain hypothesis \
if this column provides evidence
  - "needs_stage_c": true if column values need decoding \
(encoded categoricals, abbreviations, ambiguous codes)
  - "ambiguity_notes": list of concerns about this column's classification
  - "evidence": list of reasons supporting this classification
- "grain_correction": if columns contradict the grain hypothesis, \
state the correction (null otherwise)""")

    # Vocabulary family hints (domain-specific)
    if _layers.enable_vocab_hints:
        vocab_hints = build_vocab_family_hints(domain_context)
        if vocab_hints:
            parts.append(f"\n{vocab_hints}")

    # Few-shot examples (domain-specific, empty for zero-shot)
    if _layers.enable_few_shot:
        from sema.engine.few_shot import format_examples
        domain = domain_context.effective_domain if domain_context else None
        fs_block = format_examples(domain=domain, stage="B")
        if fs_block:
            parts.append(f"\n{fs_block}")

    parts.append("\nReturn ONLY valid JSON, no markdown.")

    return "\n".join(parts)


# -- Critical column identification ----------------------------------------

# -- Stage C trigger and prompt -----------------------------------------------

_STAGE_C_EXCLUDED_TYPES = frozenset({
    "identifier", "patient identifier", "encounter identifier",
    "specimen/sample identifier", "account identifier",
    "transaction identifier", "instrument identifier",
    "temporal", "temporal field", "free_text", "free text",
})

_STAGE_C_DISTINCT_THRESHOLD = 50
_STAGE_C_CONFIDENCE_THRESHOLD = 0.5


def should_trigger_stage_c(
    col: StageBColumnResult,
    col_meta: dict[str, Any] | None = None,
) -> bool:
    """Deterministic Stage C trigger per Decision 6.

    Fires if ANY of:
    1. needs_stage_c flag from B
    2. Low distinct count (≤ threshold) with top values present
    3. Low B confidence + ambiguity notes (value-driven)

    Never fires for excluded types (identifiers, temporals, free text).
    """
    if col.semantic_type.lower() in _STAGE_C_EXCLUDED_TYPES:
        return False

    # Condition 1: B explicitly flagged
    if col.needs_stage_c:
        return True

    # Condition 2: low-cardinality fallback
    if col_meta:
        distinct = col_meta.get("distinct_count")
        has_values = bool(col_meta.get("top_values"))
        if (
            distinct is not None
            and distinct <= _STAGE_C_DISTINCT_THRESHOLD
            and has_values
        ):
            return True

    # Condition 3: low confidence + ambiguity
    if (
        col.confidence < _STAGE_C_CONFIDENCE_THRESHOLD
        and col.ambiguity_notes
    ):
        return True

    return False


def build_stage_c_prompt(
    columns_with_values: list[dict[str, Any]],
    stage_a: StageAResult,
    domain_context: DomainContext | None = None,
    layers: PromptLayers | None = None,
) -> str:
    """Build Stage C prompt: value interpretation for flagged columns."""
    from sema.engine.domain_prompts import build_domain_bias_header

    _layers = layers or PromptLayers()
    parts: list[str] = []

    if _layers.enable_domain_bias:
        header = build_domain_bias_header(domain_context)
        if header:
            parts.append(header)
            parts.append("")

    parts.append(
        f"Entity context: {stage_a.primary_entity} "
        f"(grain: {stage_a.grain_hypothesis})"
    )

    parts.append("\nColumns to decode:")
    for entry in columns_with_values:
        col_name = entry["column"]
        values = entry["values"]
        vals_str = ", ".join(str(v) for v in values)
        parts.append(f"\n  Column: {col_name}")
        parts.append(f"  Values: {vals_str}")

    parts.append("""
For each column above, decode the categorical values into
human-readable meanings.

Return ONLY valid JSON with:
- "columns": array, one per column, each with:
  - "column": exact column name
  - "decoded_categories": array of {"raw": "original value", "label": "human-readable meaning"}
  - "uncertainty": 0.0–1.0 how uncertain you are about the decoding
  - "codebook_lookup_needed": true if a data dictionary would clarify ambiguous values

Return ONLY valid JSON, no markdown.""")

    # Few-shot examples for Stage C
    if _layers.enable_few_shot:
        from sema.engine.few_shot import format_examples
        domain = (
            domain_context.effective_domain if domain_context else None
        )
        fs_block = format_examples(domain=domain, stage="C")
        if fs_block:
            parts.append(f"\n{fs_block}")

    return "\n".join(parts)


def identify_critical_columns(
    column_names: list[str],
    stage_a: StageAResult,
    user_critical: set[str] | None = None,
) -> set[str]:
    """Identify Tier 1 critical columns.

    Sources: user config > key-pattern names > entity-name match.
    """
    critical: set[str] = set()
    if user_critical:
        critical.update(user_critical & set(column_names))
    for name in column_names:
        if _KEY_PATTERN.search(name):
            critical.add(name)
    entity_lower = stage_a.primary_entity.lower().replace(" ", "_")
    for name in column_names:
        if entity_lower in name.lower():
            critical.add(name)
    return critical


def classify_column_tier(
    name: str,
    critical: set[str],
    columns_meta: list[dict[str, Any]],
) -> Literal["critical", "important", "peripheral"]:
    """Assign tier for an unresolved column."""
    if name in critical:
        return "critical"
    col_meta = next(
        (c for c in columns_meta if c["name"] == name), None,
    )
    if col_meta and (col_meta.get("comment") or col_meta.get("top_values")):
        return "important"
    return "peripheral"


# -- Coverage computation --------------------------------------------------

def compute_b_coverage(
    classified: list[str], total: list[str],
) -> StageBCoverage:
    """Compute raw or critical coverage from classified vs total columns."""
    n_total = len(total)
    n_classified = len(classified)
    pct = 1.0 if n_total == 0 else round(n_classified / n_total, 4)
    return StageBCoverage(
        classified=n_classified, total=n_total, pct=pct,
    )


def determine_b_status(
    *,
    raw_coverage: StageBCoverage,
    critical_coverage: StageBCoverage,
    unresolved: list[UnresolvedColumn],
) -> Literal["B_SUCCESS", "B_PARTIAL", "B_FAILED"]:
    """Determine Stage B outcome from coverage metrics."""
    if critical_coverage.total > 0 and critical_coverage.pct < 1.0:
        return "B_FAILED"
    if raw_coverage.pct >= 1.0 and not unresolved:
        return "B_SUCCESS"
    if raw_coverage.pct >= _RAW_COVERAGE_THRESHOLD:
        return "B_PARTIAL"
    return "B_FAILED"


# -- Merge step: single assertion materialization point --------------------

def _make_assertion(
    table_ref: str,
    subject_ref: str,
    predicate: AssertionPredicate,
    payload: dict[str, Any],
    *,
    run_id: str,
    confidence: float = 0.75,
) -> Assertion:
    import uuid
    from datetime import datetime, timezone

    return Assertion(
        id=str(uuid.uuid4()),
        subject_ref=subject_ref,
        predicate=predicate,
        payload=payload,
        source="llm_interpretation",
        confidence=confidence,
        run_id=run_id,
        observed_at=datetime.now(timezone.utc),
    )


def _has_material_correction(stage_b: StageBResult) -> bool:
    """Check if B issued a material grain or entity correction."""
    return any(
        br.grain_correction is not None
        or br.entity_correction is not None
        for br in stage_b.batch_results
    )


def merge_stage_outputs(
    table_ref: str,
    stage_a: StageAResult,
    stage_b: StageBResult,
    *,
    c_results: dict[str, StageCResult] | None = None,
    run_id: str = "",
) -> list[Assertion]:
    """Single merge point: A + B + optional C → assertion list.

    Ownership (Decision 2a):
    - HAS_ENTITY_NAME: Merge(A,B) — A proposes, B corrects via grain
    - HAS_ALIAS (entity): A — dropped if B has material correction
    - HAS_PROPERTY_NAME: B exclusively
    - HAS_SEMANTIC_TYPE: B exclusively
    - HAS_ALIAS (property): B exclusively
    - HAS_DECODED_VALUE: C exclusively
    - VOCABULARY_MATCH: NOT emitted (L3 owns this)
    """
    assertions: list[Assertion] = []
    b_corrected = _has_material_correction(stage_b)

    # Entity: A proposes, B can correct both entity and grain
    entity_name = stage_a.primary_entity
    grain = stage_a.grain_hypothesis

    entity_correction = next(
        (br.entity_correction for br in stage_b.batch_results
         if br.entity_correction), None,
    )
    if entity_correction:
        entity_name = entity_correction

    grain_correction = next(
        (br.grain_correction for br in stage_b.batch_results
         if br.grain_correction), None,
    )
    if grain_correction:
        grain = grain_correction

    assertions.append(_make_assertion(
        table_ref, table_ref,
        AssertionPredicate.HAS_ENTITY_NAME,
        {"value": entity_name, "grain": grain},
        run_id=run_id,
        confidence=stage_a.confidence,
    ))

    # Entity aliases from A — dropped if B corrected entity framing
    if not b_corrected:
        for i, syn in enumerate(stage_a.synonyms):
            assertions.append(_make_assertion(
                table_ref, table_ref,
                AssertionPredicate.HAS_ALIAS,
                {"value": syn, "is_preferred": i == 0},
                run_id=run_id,
                confidence=stage_a.confidence,
            ))

    # Property-level from B (classified columns only)
    unresolved_names = {u.column for u in stage_b.unresolved_columns}
    for batch in stage_b.batch_results:
        for col in batch.columns:
            if col.column in unresolved_names:
                continue
            col_ref = f"{table_ref}.{col.column}"

            assertions.append(_make_assertion(
                table_ref, col_ref,
                AssertionPredicate.HAS_PROPERTY_NAME,
                {"value": col.canonical_property_label},
                run_id=run_id,
                confidence=col.confidence,
            ))
            assertions.append(_make_assertion(
                table_ref, col_ref,
                AssertionPredicate.HAS_SEMANTIC_TYPE,
                {"value": col.semantic_type},
                run_id=run_id,
                confidence=col.confidence,
            ))
            # Property aliases from B
            for j, syn in enumerate(col.synonyms):
                assertions.append(_make_assertion(
                    table_ref, col_ref,
                    AssertionPredicate.HAS_ALIAS,
                    {"value": syn, "is_preferred": j == 0},
                    run_id=run_id,
                    confidence=col.confidence,
                ))

    # Decoded values from C only
    if c_results:
        for col_name, c in c_results.items():
            col_ref = f"{table_ref}.{col_name}"
            for dv in c.decoded_categories:
                assertions.append(_make_assertion(
                    table_ref, col_ref,
                    AssertionPredicate.HAS_DECODED_VALUE,
                    {"raw": dv.get("raw", ""),
                     "label": dv.get("label", "")},
                    run_id=run_id,
                ))

    return assertions


# -- Enriched VocabColumnContext builder -----------------------------------

def build_enriched_vocab_context(
    col: StageBColumnResult,
    stage_a: StageAResult,
    table_name: str,
    domain_context: DomainContext | None = None,
) -> Any:
    """Build a VocabColumnContext with enrichment version 1."""
    from sema.engine.vocabulary import VocabColumnContext

    return VocabColumnContext(
        column_name=col.column,
        table_name=table_name,
        entity_name=stage_a.primary_entity,
        semantic_type=col.semantic_type,
        property_name=col.canonical_property_label,
        vocabulary_guess=None,
        vocabulary_guess_confidence=0.0,
        _enrichment_version=1,
        _candidate_vocab_families=tuple(col.candidate_vocab_families),
        _grain_hypothesis=stage_a.grain_hypothesis,
        _ambiguity_notes=tuple(col.ambiguity_notes),
        _entity_role=col.entity_role,
        _domain_context=domain_context,
    )
