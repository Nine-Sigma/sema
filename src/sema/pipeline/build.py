"""Per-table vertical processing pipeline.

Implements the discover -> enqueue -> process -> commit model.
Each table is processed through all stages sequentially on one thread,
with worker-owned resources to avoid shared mutable state.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sema.log import logger
from sema.connectors.databricks import (
    DatabricksConnector,
    TableWorkItem,
)
from sema.engine.semantic import SemanticEngine
from sema.engine.vocabulary import VocabularyEngine
from sema.graph.loader import GraphLoader
from sema.circuit_breaker import CircuitOpenError
from sema.llm_client import LLMClient, LLMStageError
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
)
from sema.models.config import (
    BuildConfig,
    DatabricksConfig,
    LLMConfig,
    ProfilingConfig,
)
from sema.models.domain import DomainContext
from sema.pipeline.build_utils import (
    _build_table_metadata,
    _commit_and_materialize,
    _count_results,
    _reconstruct_assertions,
    _run_extraction,
    _run_pipeline_stages,
    _run_semantic_interpretation,
    _run_vocabulary_alignment,
)



@dataclass
class TableResult:
    """Outcome of processing a single table."""
    table_ref: str
    status: str  # "success", "failed", "skipped"

    entities_created: int = 0
    properties_created: int = 0
    terms_created: int = 0

    failed_stage: str | None = None
    error_message: str | None = None

    skip_reason: str | None = None

    @classmethod
    def success(
        cls,
        table_ref: str,
        entities: int = 0,
        properties: int = 0,
        terms: int = 0,
    ) -> TableResult:
        return cls(
            table_ref=table_ref,
            status="success",
            entities_created=entities,
            properties_created=properties,
            terms_created=terms,
        )

    @classmethod
    def failed(
        cls, table_ref: str, stage: str, error: str
    ) -> TableResult:
        return cls(
            table_ref=table_ref,
            status="failed",
            failed_stage=stage,
            error_message=error,
        )

    @classmethod
    def skipped(cls, table_ref: str, reason: str) -> TableResult:
        return cls(
            table_ref=table_ref,
            status="skipped",
            skip_reason=reason,
        )


class DatabricksConnectorFactory:
    """Creates one DatabricksConnector per worker."""

    def __init__(
        self,
        config: DatabricksConfig,
        profiling: ProfilingConfig | None = None,
    ) -> None:
        self._config = config
        self._profiling = profiling or ProfilingConfig()

    def create(self) -> DatabricksConnector:
        return DatabricksConnector(
            config=self._config,
            profiling=self._profiling,
        )


class LLMClientFactory:
    """Creates one LLMClient per worker."""

    def __init__(
        self,
        llm_factory: Any,
        retry_max_attempts: int = 3,
        use_structured_output: str = "auto",
        circuit_breaker: Any | None = None,
    ) -> None:
        self._llm_factory = llm_factory
        self._retry_max_attempts = retry_max_attempts
        self._use_structured_output = use_structured_output
        self._circuit_breaker = circuit_breaker

    def create(self) -> LLMClient:
        llm = self._llm_factory()
        return LLMClient(
            llm=llm,
            retry_max_attempts=self._retry_max_attempts,
            use_structured_output=self._use_structured_output,
            circuit_breaker=self._circuit_breaker,
        )


def _try_resume(
    work_item: TableWorkItem,
    loader: GraphLoader,
) -> TableResult | None:
    """Check if table can be resumed from stored assertions."""
    if not loader.has_assertions(work_item.fqn):
        return None
    logger.info(f"[{work_item.table_name}] Skipping (resume): assertions exist")
    stored_dicts = loader.load_assertions(work_item.fqn)
    stored_assertions = _reconstruct_assertions(stored_dicts)
    from sema.graph.materializer import materialize_unified
    materialize_unified(loader, stored_assertions)
    return TableResult.skipped(work_item.fqn, "resume: assertions exist")


def process_table(
    work_item: TableWorkItem,
    connector: DatabricksConnector,
    llm_client: LLMClient,
    loader: GraphLoader,
    run_id: str,
    column_batch_size: int = 25,
    vocab_workers: int = 8,
    resume: bool = False,
    domain_context: DomainContext | None = None,
    use_staged: bool = False,
    prompt_layers: Any = None,
    eval_dump_dir: str | None = None,
    eval_config_label: str = "run",
) -> TableResult:
    """Process a single table through all pipeline stages."""
    if resume:
        resume_result = _try_resume(work_item, loader)
        if resume_result is not None:
            return resume_result

    try:
        result = _run_pipeline_stages(
            work_item, connector, llm_client, loader,
            run_id, column_batch_size,
            vocab_workers=vocab_workers,
            domain_context=domain_context,
            use_staged=use_staged,
            prompt_layers=prompt_layers,
        )
        if isinstance(result, TableResult):
            return result
        all_assertions, staged_output = result
    except CircuitOpenError as e:
        logger.warning(f"Table {work_item.fqn} skipped (circuit open): {e}")
        return TableResult.failed(work_item.fqn, "circuit_breaker", str(e))
    except LLMStageError as e:
        logger.warning(f"Table {work_item.fqn} failed: {e}")
        return TableResult.failed(work_item.fqn, e.stage_name, str(e))
    except Exception as e:
        logger.warning(f"Table {work_item.fqn} failed: {e}")
        return TableResult.failed(work_item.fqn, "unknown", str(e))

    if eval_dump_dir:
        _dump_for_eval(
            all_assertions, staged_output, work_item.fqn,
            eval_config_label, eval_dump_dir, run_id,
        )

    entity_count, prop_count, term_count = _count_results(all_assertions)
    return TableResult.success(
        work_item.fqn,
        entities=entity_count,
        properties=prop_count,
        terms=term_count,
    )


def _dump_for_eval(
    assertions: list[Assertion],
    staged_output: Any,
    table_ref: str,
    label: str,
    dump_dir: str,
    run_id: str,
) -> None:
    """Write eval dumps for a table. Failures are logged, not raised."""
    from pathlib import Path

    from sema.eval.pipeline_hook import (
        dump_table_eval_outputs,
        telemetry_to_dict,
    )

    try:
        telemetry = (
            telemetry_to_dict(staged_output.telemetry)
            if staged_output is not None else None
        )
        dump_table_eval_outputs(
            assertions=assertions,
            telemetry=telemetry,
            table_ref=table_ref,
            label=label,
            output_dir=Path(dump_dir),
            run_id=run_id,
        )
    except Exception as e:
        logger.warning(
            f"Eval dump failed for {table_ref}: {e}"
        )


def aggregate_report(
    results: list[TableResult],
) -> dict[str, Any]:
    """Aggregate TableResult list into a build report."""
    report: dict[str, Any] = {
        "tables_processed": 0,
        "entities_created": 0,
        "properties_created": 0,
        "terms_created": 0,
        "tables_failed": 0,
        "tables_skipped": 0,
        "failed_tables": [],
    }

    for r in results:
        if r.status == "success":
            report["tables_processed"] += 1
            report["entities_created"] += r.entities_created
            report["properties_created"] += r.properties_created
            report["terms_created"] += r.terms_created
        elif r.status == "failed":
            report["tables_failed"] += 1
            report["failed_tables"].append({
                "table": r.table_ref,
                "stage": r.failed_stage,
                "error": r.error_message,
            })
        elif r.status == "skipped":
            report["tables_skipped"] += 1

    return report
