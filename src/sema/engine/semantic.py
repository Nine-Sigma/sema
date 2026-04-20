from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

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
        llm_client: Any = None,
        run_id: str | None = None,
        column_batch_size: int = 25,
        domain_context: DomainContext | None = None,
        prompt_layers: PromptLayers | None = None,
    ) -> None:
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

    def interpret_table(
        self, table_metadata: dict[str, Any]
    ) -> list[Assertion]:
        """Interpret a single table's metadata via the staged A→B→C pipeline.

        LLMStageError propagates on Stage A failure or B_FAILED.
        """
        assertions, *_ = self.interpret_table_staged_with_metrics(
            table_metadata,
        )
        return assertions

    def interpret_tables(
        self, tables_metadata: list[dict[str, Any]]
    ) -> list[Assertion]:
        """Interpret multiple tables, returning all candidate assertions."""
        all_assertions: list[Assertion] = []
        for table in tables_metadata:
            assertions = self.interpret_table(table)
            all_assertions.extend(assertions)
        return all_assertions
