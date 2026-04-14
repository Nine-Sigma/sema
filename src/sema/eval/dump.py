"""Assertion dump capture: serialize pipeline output for comparison."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sema.models.assertions import Assertion


def dump_assertions(
    assertions: list[Assertion],
    table_ref: str,
    config_label: str,
    output_dir: Path,
    run_id: str | None = None,
) -> Path:
    """Write assertions to a JSON file for later diffing.

    Returns the path to the created file.
    """
    timestamp = datetime.now(timezone.utc)
    table_short = _extract_table_name(table_ref)
    ts_str = timestamp.strftime("%Y%m%dT%H%M%S")
    filename = f"{table_short}_{config_label}_{ts_str}.json"

    payload: dict[str, Any] = {
        "table_ref": table_ref,
        "config_label": config_label,
        "timestamp": timestamp.isoformat(),
        "run_id": run_id,
        "assertions": [_serialize_assertion(a) for a in assertions],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path


def load_dump(path: Path) -> dict[str, Any]:
    """Load a previously-saved assertion dump."""
    if not path.exists():
        msg = f"Dump file not found: {path}"
        raise FileNotFoundError(msg)
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _serialize_assertion(a: Assertion) -> dict[str, Any]:
    """Serialize an assertion to a diffable dict."""
    return {
        "subject_ref": a.subject_ref,
        "predicate": a.predicate.value,
        "payload": a.payload,
        "confidence": a.confidence,
        "source": a.source,
    }


def _extract_table_name(table_ref: str) -> str:
    """Extract short table name from a table ref like 'unity://cat.sch.table'."""
    parts = table_ref.rstrip("/").split(".")
    return parts[-1] if parts else table_ref
