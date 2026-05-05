"""Forensic artifact persistence for failed tables.

Writes `{table}__{label}__failure.json` alongside the existing
`*__telemetry.json` artifacts so post-mortem can be done offline.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sema.eval.failure_artifact_utils import (
    counters_for_exc,
    llm_attempts_for_exc,
    prompt_hashes_for_exc,
    stage_a_output_for_exc,
    step_errors_for_exc,
    unresolved_columns_for_exc,
)
from sema.eval.runner_utils import dump_filename, write_json


def classify_failure(exc: BaseException) -> str:
    from sema.circuit_breaker import CircuitOpenError
    from sema.llm_client import (
        LLMStageError,
        StageBFailureError,
        _is_rate_limit_error,
        _is_service_health_failure,
    )

    if isinstance(exc, CircuitOpenError):
        return "circuit_open"
    if isinstance(exc, StageBFailureError):
        return "semantic_coverage"
    if isinstance(exc, LLMStageError):
        if any(_is_service_health_failure(e) for _, e in exc.step_errors):
            return "service_health"
        if exc.step_errors and all(
            _is_rate_limit_error(e) for _, e in exc.step_errors
        ):
            return "rate_limit"
        if not exc.step_errors:
            return "unknown"
        return "content_failure"
    return "unknown"


def build_failure_artifact(
    *,
    exc: BaseException,
    table_ref: str,
    run_id: str,
    failure_stage: str,
    metadata_tier: str | None = None,
) -> dict[str, Any]:
    return {
        "table_ref": table_ref,
        "run_id": run_id,
        "failure_stage": failure_stage,
        "failure_classification": classify_failure(exc),
        "metadata_tier": metadata_tier,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error_message": str(exc),
        "step_errors": step_errors_for_exc(exc),
        "stage_a_output": stage_a_output_for_exc(exc),
        "unresolved_columns": unresolved_columns_for_exc(exc),
        "counters": counters_for_exc(exc),
        "prompt_hashes": prompt_hashes_for_exc(exc),
        "llm_attempts": llm_attempts_for_exc(exc),
    }


def failure_dump_filename(table_ref: str, label: str) -> str:
    base = dump_filename(table_ref, label, telemetry=False)
    return base.removesuffix(".json") + "__failure.json"


def dump_table_failure_artifact(
    *,
    exc: BaseException,
    table_ref: str,
    label: str,
    output_dir: Path | str | None,
    run_id: str,
    failure_stage: str,
    metadata_tier: str | None = None,
) -> Path | None:
    if output_dir is None:
        return None
    out_dir = Path(output_dir)
    artifact = build_failure_artifact(
        exc=exc,
        table_ref=table_ref,
        run_id=run_id,
        failure_stage=failure_stage,
        metadata_tier=metadata_tier,
    )
    out = out_dir / failure_dump_filename(table_ref, label)
    return write_json(out, artifact)
