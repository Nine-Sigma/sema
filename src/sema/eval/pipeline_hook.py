"""Bridge between per-table pipeline output and eval dump files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from sema.eval.runner import write_table_dump, write_telemetry_dump
from sema.models.assertions import Assertion


def dump_table_eval_outputs(
    *,
    assertions: list[Assertion],
    telemetry: dict[str, Any] | None,
    table_ref: str,
    label: str,
    output_dir: Path,
    run_id: str | None = None,
) -> None:
    """Write per-table assertion and (optional) telemetry dumps."""
    write_table_dump(
        assertions,
        table_ref=table_ref,
        label=label,
        output_dir=output_dir,
        run_id=run_id,
    )
    if telemetry is not None:
        write_telemetry_dump(
            telemetry,
            table_ref=table_ref,
            label=label,
            output_dir=output_dir,
        )


def telemetry_to_dict(telemetry: Any) -> dict[str, Any]:
    """Convert TableTelemetry dataclass to a JSON-serializable dict."""
    if telemetry is None:
        return {}
    return {
        "table_ref": telemetry.table_ref,
        "stage_a_calls": telemetry.stage_a_calls,
        "stage_b_batches_attempted": telemetry.stage_b_batches_attempted,
        "stage_b_batches_succeeded": telemetry.stage_b_batches_succeeded,
        "stage_c_calls": telemetry.stage_c_calls,
        "b_outcome": telemetry.b_outcome,
        "retries_used": telemetry.retries_used,
        "splits_used": telemetry.splits_used,
        "rescues_used": telemetry.rescues_used,
        "raw_coverage_pct": telemetry.raw_coverage_pct,
        "critical_coverage_pct": telemetry.critical_coverage_pct,
        "c_columns_flagged": telemetry.c_columns_flagged,
        "total_columns": telemetry.total_columns,
        "c_trigger_rate": telemetry.c_trigger_rate,
        "stage_a_latency_ms": telemetry.stage_a_latency_ms,
        "stage_b_latency_ms": telemetry.stage_b_latency_ms,
        "stage_c_latency_ms": telemetry.stage_c_latency_ms,
        "total_latency_ms": telemetry.total_latency_ms,
        "tokens_input": telemetry.tokens_input,
        "tokens_output": telemetry.tokens_output,
    }
