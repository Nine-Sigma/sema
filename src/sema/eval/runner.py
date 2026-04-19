"""Eval slice runner: loads slice defs, writes per-table dumps, builds reports."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sema.eval.diff import diff_dumps
from sema.eval.dump import _serialize_assertion
from sema.eval.runner_utils import (
    SliceDefinition,
    dump_filename,
    list_assertion_dumps,
    list_telemetry_dumps,
    load_json,
    parse_slice_yaml,
    write_json,
)
from sema.models.assertions import Assertion


def load_slice(path: Path) -> SliceDefinition:
    """Load a slice YAML file into a SliceDefinition."""
    if not path.exists():
        msg = f"Slice file not found: {path}"
        raise FileNotFoundError(msg)

    data = yaml.safe_load(path.read_text()) or {}
    return parse_slice_yaml(data)


def write_table_dump(
    assertions: list[Assertion],
    *,
    table_ref: str,
    label: str,
    output_dir: Path,
    run_id: str | None = None,
) -> Path:
    """Write a per-table assertion dump with a deterministic filename."""
    payload = {
        "table_ref": table_ref,
        "config_label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "assertions": [_serialize_assertion(a) for a in assertions],
    }
    out = output_dir / dump_filename(table_ref, label, telemetry=False)
    return write_json(out, payload)


def write_telemetry_dump(
    telemetry: dict[str, Any],
    *,
    table_ref: str,
    label: str,
    output_dir: Path,
) -> Path:
    """Write a per-table telemetry dump alongside the assertion dump."""
    out = output_dir / dump_filename(table_ref, label, telemetry=True)
    return write_json(out, telemetry)


def pair_dumps_by_table(
    baseline_dir: Path, current_dir: Path,
) -> tuple[list[tuple[str, Path, Path]], dict[str, list[str]]]:
    """Pair assertion dumps by short table name across two directories.

    Returns (pairs, unmatched) where pairs is a list of
    (table_short, baseline_path, current_path) and unmatched is
    {"only_in_baseline": [...], "only_in_current": [...]}.
    """
    baseline = list_assertion_dumps(baseline_dir)
    current = list_assertion_dumps(current_dir)
    shared = sorted(set(baseline.keys()) & set(current.keys()))
    pairs = [(t, baseline[t], current[t]) for t in shared]
    unmatched = {
        "only_in_baseline": sorted(
            set(baseline.keys()) - set(current.keys()),
        ),
        "only_in_current": sorted(
            set(current.keys()) - set(baseline.keys()),
        ),
    }
    return pairs, unmatched


def build_diff_report(
    baseline_dir: Path, current_dir: Path,
) -> dict[str, Any]:
    """Diff every paired table dump, aggregate churn summary."""
    pairs, unmatched = pair_dumps_by_table(baseline_dir, current_dir)
    per_table: list[dict[str, Any]] = []
    totals = {"added": 0, "removed": 0, "changed": 0}

    for table_short, bpath, cpath in pairs:
        bdump = load_json(bpath)
        cdump = load_json(cpath)
        diff = diff_dumps(bdump, cdump)
        per_table.append({
            "table": table_short,
            "summary": diff["summary"],
        })
        totals["added"] += diff["summary"]["added_count"]
        totals["removed"] += diff["summary"]["removed_count"]
        totals["changed"] += diff["summary"]["changed_count"]

    return {
        "summary": {
            "tables_compared": len(pairs),
            "total_added": totals["added"],
            "total_removed": totals["removed"],
            "total_changed": totals["changed"],
            "only_in_baseline": unmatched["only_in_baseline"],
            "only_in_current": unmatched["only_in_current"],
        },
        "per_table": per_table,
    }


def build_run_report(
    run_dir: Path,
    *,
    label: str,
    baseline_dir: Path | None = None,
) -> dict[str, Any]:
    """Aggregate telemetry dumps into a milestone-style report."""
    telemetry_paths = list_telemetry_dumps(run_dir)
    loaded = [load_json(p) for p in telemetry_paths]

    telemetry_summary = _aggregate_telemetry(loaded)
    report: dict[str, Any] = {
        "label": label,
        "telemetry": telemetry_summary,
    }
    if baseline_dir is not None and baseline_dir.exists():
        report["semantic_churn"] = build_diff_report(
            baseline_dir, run_dir,
        )["summary"]
    return report


def _aggregate_telemetry(
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    if not items:
        return {
            "table_count": 0,
            "b_outcome_distribution": {
                "success": 0, "partial": 0, "failed": 0,
            },
        }
    n = len(items)
    dist = {"success": 0, "partial": 0, "failed": 0}
    for item in items:
        outcome = item.get("b_outcome", "")
        if outcome == "B_SUCCESS":
            dist["success"] += 1
        elif outcome == "B_PARTIAL":
            dist["partial"] += 1
        elif outcome == "B_FAILED":
            dist["failed"] += 1

    def _avg(field: str) -> float:
        vals = [float(it.get(field, 0) or 0) for it in items]
        return sum(vals) / n if vals else 0.0

    return {
        "table_count": n,
        "b_outcome_distribution": dist,
        "avg_raw_coverage_pct": round(_avg("raw_coverage_pct"), 4),
        "avg_critical_coverage_pct": round(
            _avg("critical_coverage_pct"), 4,
        ),
        "avg_c_trigger_rate": round(_avg("c_trigger_rate"), 4),
        "avg_total_latency_ms": round(_avg("total_latency_ms"), 1),
        "recovery": {
            "total_retries": sum(
                int(it.get("retries_used", 0) or 0) for it in items
            ),
            "total_splits": sum(
                int(it.get("splits_used", 0) or 0) for it in items
            ),
            "total_rescues": sum(
                int(it.get("rescues_used", 0) or 0) for it in items
            ),
        },
        "tokens": {
            "input": sum(
                int(it.get("tokens_input", 0) or 0) for it in items
            ),
            "output": sum(
                int(it.get("tokens_output", 0) or 0) for it in items
            ),
        },
    }
