from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from sema.engine.semantic_utils import (
    entity_assertions,
    property_assertions,
    run_property_pass,
    run_summary_pass,
)
from sema.engine.stage_utils import (
    PromptLayers,
    build_stage_a_prompt,
    build_stage_b_prompt,
    build_stage_c_prompt,
    classify_column_tier,
    compute_b_coverage,
    determine_b_status,
    identify_critical_columns,
    merge_stage_outputs,
    sanitize_column_name,
    should_trigger_stage_c,
)
from sema.llm_client import LLMStageError
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
)
from sema.models.domain import DomainContext
from sema.models.stages import (
    StageAResult,
    StageBBatchResult,
    StageBColumnResult,
    StageBResult,
    StageCBatchResult,
    StageCResult,
    UnresolvedColumn,
)

logger = logging.getLogger(__name__)


class DecodedValue(BaseModel):
    raw: str
    label: str


class PropertyInterpretation(BaseModel):
    column: str
    name: str
    description: str | None = None
    semantic_type: str
    vocabulary_guess: str | None = None
    confidence: float = 0.75
    synonyms: list[str] | None = None
    decoded_values: list[dict[str, str]] | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.synonyms is None:
            self.synonyms = []
        if self.decoded_values is None:
            self.decoded_values = []


class TableInterpretation(BaseModel):
    entity_name: str
    entity_description: str | None = None
    synonyms: list[str] | None = None
    properties: list[PropertyInterpretation] | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.synonyms is None:
            self.synonyms = []
        if self.properties is None:
            self.properties = []


def build_interpretation_prompt(
    table_metadata: dict[str, Any],
    max_sample_values: int = 10,
) -> str:
    """Build the LLM prompt for interpreting a table's metadata."""
    parts = [
        f"Table: {table_metadata['table_name']}",
    ]
    if table_metadata.get("comment"):
        parts.append(f"Comment: {table_metadata['comment']}")

    parts.append("\nColumns:")
    for col in table_metadata.get("columns", []):
        line = f"  {col['name']} ({col['data_type']})"
        if col.get("comment"):
            line += f" - {col['comment']}"
        if col.get("top_values"):
            values = [v["value"] for v in col["top_values"][:max_sample_values]]
            line += f"\n    top values: {', '.join(values)}"
        parts.append(line)

    if table_metadata.get("sample_rows"):
        parts.append("\nSample rows:")
        for row in table_metadata["sample_rows"][:5]:
            parts.append(f"  {json.dumps(row)}")

    parts.append("""
Generate a JSON object with:
1. "entity_name": human-readable concept name for this table
2. "entity_description": what this table represents
3. "synonyms": alternative names someone might search for
4. "properties": array, one per column, each with:
   - "column": exact column name
   - "name": human-readable property name
   - "description": what this column means
   - "semantic_type": one of "identifier", "categorical", "temporal", "numeric", "free_text"
   - "vocabulary_guess": if categorical, what standard vocabulary (e.g., ICD-10, OncoTree, AJCC)
   - "confidence": 0.0-1.0 how confident you are
   - "synonyms": alternative names for this property
   - "decoded_values": if categorical, array of {"raw": "code", "label": "human name"}

Return ONLY valid JSON, no markdown.
""")

    return "\n".join(parts)


def build_simplified_interpretation_prompt(
    table_metadata: dict[str, Any],
) -> str:
    """Build a simplified prompt requesting TableInterpretation with minimal data.

    Sends only column names and types (no values or comments).
    Requests entity_name, entity_description, synonyms, and properties
    with column, name, and semantic_type only.
    """
    parts = [f"Table: {table_metadata['table_name']}"]

    parts.append("\nColumns (name : type):")
    for col in table_metadata.get("columns", []):
        parts.append(f"  {col['name']} : {col['data_type']}")

    parts.append("""
Return ONLY valid JSON with:
- "entity_name": human-readable concept name
- "entity_description": one-sentence description
- "synonyms": alternative names
- "properties": array, one per column, each with:
  - "column": exact column name
  - "name": human-readable property name
  - "semantic_type": one of "identifier", "categorical", "temporal", "numeric", "free_text"
""")
    return "\n".join(parts)


def build_summary_prompt(table_metadata: dict[str, Any]) -> str:
    """Build a lightweight prompt for the table summary pass.

    Sends only column names and types — no values, no descriptions.
    Asks for entity name, description, and synonyms only.
    """
    parts = [f"Table: {table_metadata['table_name']}"]
    if table_metadata.get("comment"):
        parts.append(f"Comment: {table_metadata['comment']}")

    parts.append("\nColumns (name : type):")
    for col in table_metadata.get("columns", []):
        parts.append(f"  {col['name']} : {col['data_type']}")

    parts.append("""
Based on the table name and column names above, determine what business
entity this table represents.

Return ONLY valid JSON with:
- "entity_name": human-readable concept name
- "entity_description": one-sentence description
- "synonyms": alternative names someone might search for
""")
    return "\n".join(parts)


def build_property_prompt(
    table_metadata: dict[str, Any],
    columns: list[dict[str, Any]],
    entity_name: str,
    max_sample_values: int = 10,
) -> str:
    """Build a prompt for a chunked property extraction pass."""
    parts = [
        f"Table: {table_metadata['table_name']}",
        f"This table represents: {entity_name}",
        "\nColumns to interpret:",
    ]
    for col in columns:
        line = f"  {col['name']} ({col['data_type']})"
        if col.get("comment"):
            line += f" - {col['comment']}"
        if col.get("top_values"):
            values = [
                v["value"] for v in col["top_values"][:max_sample_values]
            ]
            line += f"\n    top values: {', '.join(values)}"
        parts.append(line)

    parts.append("""
For each column above, return a JSON object with:
- "properties": array, one per column, each with:
  - "column": exact column name
  - "name": human-readable property name
  - "description": what this column means
  - "semantic_type": one of "identifier", "categorical", "temporal", "numeric", "free_text"
  - "vocabulary_guess": if categorical, what standard vocabulary
  - "confidence": 0.0-1.0
  - "synonyms": alternative names
  - "decoded_values": if categorical, array of {"raw": "code", "label": "human name"}

Return ONLY valid JSON, no markdown.
""")
    return "\n".join(parts)


class _PropertyBatchResult(BaseModel):
    properties: list[PropertyInterpretation] = []


@dataclass
class StageMetrics:
    """Per-table timing and token accounting across Stage A/B/C."""

    stage_a_latency_ms: int = 0
    stage_b_latency_ms: int = 0
    stage_c_latency_ms: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    stage_c_calls: int = 0


class SemanticEngine:
    """L2: LLM-assisted semantic interpretation of table metadata."""

    def __init__(
        self,
        llm: Any = None,
        run_id: str | None = None,
        llm_client: Any = None,
        column_batch_size: int = 25,
        domain_context: DomainContext | None = None,
        prompt_layers: PromptLayers | None = None,
    ) -> None:
        self._llm = llm
        self._llm_client = llm_client
        self._run_id = run_id or str(uuid.uuid4())
        self._column_batch_size = column_batch_size
        self._domain_context = domain_context
        self._layers = prompt_layers or PromptLayers()

    def run_stage_a(
        self, table_metadata: dict[str, Any],
    ) -> StageAResult:
        """Stage A: entity and grain hypothesis.

        Returns typed intermediate — does NOT produce assertions.
        LLMStageError propagates to caller on failure.
        """
        table_ref = table_metadata.get(
            "table_ref",
            f"unity://{table_metadata.get('table_name', 'unknown')}",
        )
        prompt = build_stage_a_prompt(
            table_metadata, domain_context=self._domain_context,
            layers=self._layers,
        )
        return self._llm_client.invoke(  # type: ignore[no-any-return]
            prompt,
            StageAResult,
            table_ref=table_ref,
            stage_name="L2 stage_a",
        )

    def _invoke_stage_b_batch(
        self,
        table_metadata: dict[str, Any],
        batch: list[dict[str, Any]],
        stage_a: StageAResult,
        table_ref: str,
    ) -> StageBBatchResult:
        prompt = build_stage_b_prompt(
            table_metadata, batch, stage_a,
            domain_context=self._domain_context,
            layers=self._layers,
        )
        result: StageBBatchResult = self._llm_client.invoke(
            prompt,
            StageBBatchResult,
            table_ref=table_ref,
            stage_name="L2 stage_b",
        )
        for c in result.columns:
            c.column = sanitize_column_name(c.column)
        return result

    def _run_batch_with_recovery(
        self,
        table_metadata: dict[str, Any],
        batch: list[dict[str, Any]],
        stage_a: StageAResult,
        table_ref: str,
    ) -> tuple[list[StageBBatchResult], list[dict[str, Any]], int, int]:
        """Run a batch with bounded recovery: retry once, split once.

        Returns (results, failed_cols, retries_used, splits_used).
        """
        retries = 0
        splits = 0

        try:
            result = self._invoke_stage_b_batch(
                table_metadata, batch, stage_a, table_ref,
            )
            return [result], [], retries, splits
        except LLMStageError:
            pass

        # Retry once
        retries += 1
        try:
            result = self._invoke_stage_b_batch(
                table_metadata, batch, stage_a, table_ref,
            )
            return [result], [], retries, splits
        except LLMStageError:
            pass

        # Split into two halves — no further recovery on sub-batches
        if len(batch) < 2:
            return [], batch, retries, splits

        splits += 1
        mid = len(batch) // 2
        results: list[StageBBatchResult] = []
        failed_cols: list[dict[str, Any]] = []
        for half in (batch[:mid], batch[mid:]):
            try:
                r = self._invoke_stage_b_batch(
                    table_metadata, half, stage_a, table_ref,
                )
                results.append(r)
            except LLMStageError:
                failed_cols.extend(half)
        return results, failed_cols, retries, splits

    def _rescue_critical_columns(
        self,
        table_metadata: dict[str, Any],
        failed_cols: list[dict[str, Any]],
        critical: set[str],
        stage_a: StageAResult,
        table_ref: str,
    ) -> tuple[list[StageBBatchResult], list[dict[str, Any]]]:
        """Optional Tier 1 rescue: one call for unresolved critical cols."""
        crit_batch = [c for c in failed_cols if c["name"] in critical]
        if not crit_batch:
            return [], failed_cols
        try:
            result = self._invoke_stage_b_batch(
                table_metadata, crit_batch, stage_a, table_ref,
            )
            remaining = [c for c in failed_cols if c["name"] not in critical]
            return [result], remaining
        except LLMStageError:
            return [], failed_cols

    def run_stage_b(
        self,
        table_metadata: dict[str, Any],
        stage_a: StageAResult,
    ) -> StageBResult:
        """Stage B: property classification with batching and recovery.

        Returns typed intermediate — does NOT produce assertions.
        """
        table_ref = table_metadata.get(
            "table_ref",
            f"unity://{table_metadata.get('table_name', 'unknown')}",
        )
        columns = table_metadata.get("columns", [])
        all_col_names = [c["name"] for c in columns]
        critical = identify_critical_columns(all_col_names, stage_a)

        all_batches: list[StageBBatchResult] = []
        all_failed: list[dict[str, Any]] = []
        total_retries = 0
        total_splits = 0
        rescues_used = 0

        for i in range(0, len(columns), self._column_batch_size):
            batch = columns[i:i + self._column_batch_size]
            results, failed, retries, splits = (
                self._run_batch_with_recovery(
                    table_metadata, batch, stage_a, table_ref,
                )
            )
            all_batches.extend(results)
            all_failed.extend(failed)
            total_retries += retries
            total_splits += splits

        # Optional Tier 1 rescue for unresolved critical columns
        crit_failed = [c for c in all_failed if c["name"] in critical]
        if crit_failed:
            rescued, remaining = self._rescue_critical_columns(
                table_metadata, all_failed, critical, stage_a, table_ref,
            )
            if rescued:
                all_batches.extend(rescued)
                all_failed = remaining
                rescues_used = 1

        # Downgrade low-confidence ambiguous results to unresolved
        _SEMANTIC_CONFIDENCE_FLOOR = 0.4
        semantic_unresolved_names: set[str] = set()
        for br in all_batches:
            for cr in br.columns:
                if (
                    cr.confidence < _SEMANTIC_CONFIDENCE_FLOOR
                    and cr.ambiguity_notes
                ):
                    semantic_unresolved_names.add(cr.column)

        classified = [
            cr.column
            for br in all_batches
            for cr in br.columns
            if cr.column not in semantic_unresolved_names
        ]
        failed_names = [c["name"] for c in all_failed]

        unresolved = [
            UnresolvedColumn(
                column=name,
                reason="execution_failure",
                tier=classify_column_tier(name, critical, columns),
            )
            for name in failed_names
        ]
        unresolved.extend(
            UnresolvedColumn(
                column=name,
                reason="semantic_unresolved",
                tier=classify_column_tier(name, critical, columns),
            )
            for name in sorted(semantic_unresolved_names)
        )

        raw_cov = compute_b_coverage(classified, all_col_names)
        crit_classified = [n for n in classified if n in critical]
        crit_total = [n for n in all_col_names if n in critical]
        crit_cov = compute_b_coverage(crit_classified, crit_total)

        status = determine_b_status(
            raw_coverage=raw_cov,
            critical_coverage=crit_cov,
            unresolved=unresolved,
        )

        return StageBResult(
            status=status,
            batch_results=all_batches,
            raw_coverage=raw_cov,
            critical_coverage=crit_cov,
            unresolved_columns=unresolved,
            retries_used=total_retries,
            splits_used=total_splits,
            rescues_used=rescues_used,
        )

    def run_stage_c(
        self,
        table_metadata: dict[str, Any],
        stage_a: StageAResult,
        stage_b: StageBResult,
    ) -> dict[str, StageCResult]:
        """Stage C: conditional value interpretation for flagged columns.

        Returns a dict of column_name → StageCResult.
        Skips unresolved B columns and excluded types.
        Returns empty dict when enable_stage_c is False.
        Partial failures: successful results returned, failures logged.
        """
        if not self._layers.enable_stage_c:
            return {}

        table_ref = table_metadata.get(
            "table_ref",
            f"unity://{table_metadata.get('table_name', 'unknown')}",
        )
        unresolved = {u.column for u in stage_b.unresolved_columns}

        # Build column metadata index for trigger decisions
        col_meta_index: dict[str, dict[str, Any]] = {
            cm["name"]: cm
            for cm in table_metadata.get("columns", [])
        }

        # Collect columns eligible for Stage C
        eligible: list[StageBColumnResult] = []
        for batch in stage_b.batch_results:
            for col in batch.columns:
                if col.column in unresolved:
                    continue
                cm = col_meta_index.get(col.column)
                if should_trigger_stage_c(col, col_meta=cm):
                    eligible.append(col)

        if not eligible:
            return {}

        # Build value map from table metadata
        col_values: dict[str, list[str]] = {}
        for cm in table_metadata.get("columns", []):
            if cm.get("top_values"):
                col_values[cm["name"]] = [
                    v["value"] for v in cm["top_values"]
                ]

        # Build batched prompt input for eligible columns
        prompt_input = [
            {"column": col.column, "values": col_values[col.column]}
            for col in eligible
            if col.column in col_values and col_values[col.column]
        ]
        if not prompt_input:
            return {}

        results: dict[str, StageCResult] = {}
        prompt = build_stage_c_prompt(
            prompt_input, stage_a,
            domain_context=self._domain_context,
            layers=self._layers,
        )
        try:
            batch_result = self._llm_client.invoke(
                prompt,
                StageCBatchResult,
                table_ref=table_ref,
                stage_name="L2 stage_c",
            )
            for cr in batch_result.columns:
                results[cr.column] = cr
        except LLMStageError:
            # Fallback: try per-column on batch failure
            for entry in prompt_input:
                col_name = entry["column"]
                single_prompt = build_stage_c_prompt(
                    [entry], stage_a,
                    domain_context=self._domain_context,
                    layers=self._layers,
                )
                try:
                    cr = self._llm_client.invoke(
                        single_prompt,
                        StageCResult,
                        table_ref=table_ref,
                        stage_name="L2 stage_c",
                    )
                    results[cr.column] = cr
                except LLMStageError:
                    logger.warning(
                        f"Stage C failed for {col_name} "
                        f"in {table_ref}"
                    )
        return results

    def interpret_table_staged(
        self, table_metadata: dict[str, Any],
    ) -> tuple[
        list[Assertion], StageAResult, StageBResult,
        dict[str, StageCResult],
    ]:
        """Staged A→B→C→merge pipeline (legacy signature).

        Returns (assertions, stage_a, stage_b, c_results).
        Use `interpret_table_staged_with_metrics` for timing/tokens.
        """
        assertions, a, b, c, _ = (
            self.interpret_table_staged_with_metrics(table_metadata)
        )
        return assertions, a, b, c

    def interpret_table_staged_with_metrics(
        self, table_metadata: dict[str, Any],
    ) -> tuple[
        list[Assertion], StageAResult, StageBResult,
        dict[str, StageCResult], StageMetrics,
    ]:
        """Same as `interpret_table_staged` plus per-stage metrics."""
        import time

        metrics = StageMetrics()
        original_invoke = self._llm_client.invoke

        def _wrapped_invoke(*args: Any, **kwargs: Any) -> Any:
            result = original_invoke(*args, **kwargs)
            stats = getattr(self._llm_client, "last_stats", None)
            if stats is not None:
                metrics.tokens_input += int(stats.prompt_tokens)
                metrics.tokens_output += int(stats.completion_tokens)
            return result

        self._llm_client.invoke = _wrapped_invoke
        try:
            start = time.monotonic_ns()
            stage_a = self.run_stage_a(table_metadata)
            metrics.stage_a_latency_ms = (
                (time.monotonic_ns() - start) // 1_000_000
            )

            start = time.monotonic_ns()
            stage_b = self.run_stage_b(table_metadata, stage_a)
            metrics.stage_b_latency_ms = (
                (time.monotonic_ns() - start) // 1_000_000
            )

            if stage_b.status == "B_FAILED":
                table_ref = table_metadata.get(
                    "table_ref",
                    f"unity://{table_metadata.get('table_name', 'unknown')}",
                )
                raise LLMStageError(
                    table_ref=table_ref,
                    stage_name="L2 stage_b",
                    step_errors=[("stage_b", ValueError(
                        f"B_FAILED: raw={stage_b.raw_coverage.pct}"
                    ))],
                )

            start = time.monotonic_ns()
            c_results = self.run_stage_c(
                table_metadata, stage_a, stage_b,
            )
            metrics.stage_c_latency_ms = (
                (time.monotonic_ns() - start) // 1_000_000
            )
            metrics.stage_c_calls = len(c_results)
        finally:
            self._llm_client.invoke = original_invoke

        table_ref = table_metadata.get(
            "table_ref",
            f"unity://{table_metadata.get('table_name', 'unknown')}",
        )
        assertions = merge_stage_outputs(
            table_ref, stage_a, stage_b,
            c_results=c_results,
            run_id=self._run_id,
        )
        return assertions, stage_a, stage_b, c_results, metrics

    def _make_assertion(
        self,
        subject_ref: str,
        predicate: AssertionPredicate,
        payload: dict[str, Any],
        object_ref: str | None = None,
        confidence: float = 0.75,
    ) -> Assertion:
        return Assertion(
            id=str(uuid.uuid4()),
            subject_ref=subject_ref,
            predicate=predicate,
            payload=payload,
            object_ref=object_ref,
            source="llm_interpretation",
            confidence=confidence,
            run_id=self._run_id,
            observed_at=datetime.now(timezone.utc),
        )

    def _interpret_via_llm_client(
        self, table_metadata: dict[str, Any], table_ref: str
    ) -> TableInterpretation:
        prompt = build_interpretation_prompt(table_metadata)
        simplified = build_simplified_interpretation_prompt(table_metadata)
        return self._llm_client.invoke(  # type: ignore[no-any-return]
            prompt,
            TableInterpretation,
            table_ref=table_ref,
            stage_name="L2 semantic",
            simplified_prompt=simplified,
        )

    def _interpret_via_raw_llm(
        self, table_metadata: dict[str, Any], table_ref: str
    ) -> TableInterpretation:
        prompt = build_interpretation_prompt(table_metadata)
        try:
            response = self._llm.invoke(prompt)
            raw_content = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
            raw_content = raw_content.strip()
            if raw_content.startswith("```"):
                lines = raw_content.split("\n")
                lines = [
                    line
                    for line in lines
                    if not line.strip().startswith("```")
                ]
                raw_content = "\n".join(lines).strip()
            return TableInterpretation.model_validate_json(raw_content)
        except Exception as e:
            logger.warning(
                f"LLM interpretation failed for {table_ref}: {e}"
            )
            return None  # type: ignore[return-value]

    def _needs_two_pass(self, table_metadata: dict[str, Any]) -> bool:
        columns = table_metadata.get("columns", [])
        threshold = self._column_batch_size * 2
        return len(columns) >= threshold and self._llm_client is not None

    def _run_summary_pass(
        self, table_metadata: dict[str, Any], table_ref: str
    ) -> tuple[list[Assertion], Any]:
        return run_summary_pass(self, table_metadata, table_ref)

    def _run_property_pass(
        self, table_metadata: dict[str, Any], table_ref: str, entity_name: str
    ) -> list[Assertion]:
        return run_property_pass(self, table_metadata, table_ref, entity_name)

    def _interpret_two_pass(
        self, table_metadata: dict[str, Any], table_ref: str
    ) -> list[Assertion]:
        summary_assertions, summary = self._run_summary_pass(table_metadata, table_ref)
        property_assertions_list = self._run_property_pass(table_metadata, table_ref, summary.entity_name)
        return summary_assertions + property_assertions_list

    def interpret_table(
        self, table_metadata: dict[str, Any]
    ) -> list[Assertion]:
        """Interpret a single table's metadata via LLM.

        Uses two-pass strategy for wide tables (>= 2*column_batch_size cols).
        When using LLMClient: LLMStageError propagates to caller (no catch).
        When using raw LLM (legacy): errors are swallowed and empty list returned.
        """
        table_ref = table_metadata.get(
            "table_ref",
            f"unity://{table_metadata.get('table_name', 'unknown')}",
        )

        if self._needs_two_pass(table_metadata):
            return self._interpret_two_pass(table_metadata, table_ref)

        if self._llm_client:
            # New path: LLMStageError propagates — no catch
            interpretation = self._interpret_via_llm_client(
                table_metadata, table_ref
            )
        elif self._llm:
            # Legacy path: swallows errors
            interpretation = self._interpret_via_raw_llm(
                table_metadata, table_ref
            )
            if interpretation is None:
                return []
        else:
            return []

        return self._interpretation_to_assertions(interpretation, table_ref)

    def _entity_assertions(
        self, interpretation: TableInterpretation, table_ref: str
    ) -> list[Assertion]:
        return entity_assertions(self, interpretation, table_ref)

    def _property_assertions(
        self, interpretation: TableInterpretation, table_ref: str
    ) -> list[Assertion]:
        return property_assertions(self, interpretation, table_ref)

    def _interpretation_to_assertions(
        self, interpretation: TableInterpretation, table_ref: str
    ) -> list[Assertion]:
        return (
            self._entity_assertions(interpretation, table_ref)
            + self._property_assertions(interpretation, table_ref)
        )

    def interpret_tables(
        self, tables_metadata: list[dict[str, Any]]
    ) -> list[Assertion]:
        """Interpret multiple tables, returning all candidate assertions."""
        all_assertions: list[Assertion] = []
        for table in tables_metadata:
            assertions = self.interpret_table(table)
            all_assertions.extend(assertions)
        return all_assertions
