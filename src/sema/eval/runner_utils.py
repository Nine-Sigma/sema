"""Helpers for eval slice runner: parsing, dump I/O, pairing, aggregation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SliceDefinition:
    """A dev slice or holdout definition loaded from YAML."""

    catalog: str
    schema: str
    tables: list[str]
    table_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    version: int = 1


def parse_slice_yaml(data: dict[str, Any]) -> SliceDefinition:
    tables_raw = data.get("tables", [])
    if not tables_raw:
        msg = "slice has no tables"
        raise ValueError(msg)

    tables: list[str] = []
    table_meta: dict[str, dict[str, Any]] = {}
    for entry in tables_raw:
        name = entry["table_name"]
        tables.append(name)
        table_meta[name] = {
            k: v for k, v in entry.items() if k != "table_name"
        }

    return SliceDefinition(
        catalog=data.get("catalog", ""),
        schema=data.get("schema", ""),
        tables=tables,
        table_meta=table_meta,
        version=int(data.get("version", 1)),
    )


def extract_table_name(table_ref: str) -> str:
    """Extract short table name from a ref like 'unity://c/s/t' or 'a.b.c'."""
    if "/" in table_ref:
        return table_ref.rstrip("/").split("/")[-1]
    parts = table_ref.split(".")
    return parts[-1]


def dump_filename(table_ref: str, label: str, *, telemetry: bool) -> str:
    short = extract_table_name(table_ref)
    suffix = "__telemetry" if telemetry else ""
    return f"{short}__{label}{suffix}.json"


def list_assertion_dumps(run_dir: Path) -> dict[str, Path]:
    """Return a {table_short: path} map, excluding telemetry files."""
    result: dict[str, Path] = {}
    for p in run_dir.glob("*.json"):
        if p.stem.endswith("__telemetry"):
            continue
        short = p.stem.split("__", 1)[0]
        result[short] = p
    return result


def list_telemetry_dumps(run_dir: Path) -> list[Path]:
    return [p for p in run_dir.glob("*__telemetry.json")]


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data
