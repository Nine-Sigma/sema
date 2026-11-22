"""Hashing and serialization helpers for the run ledger."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

PROMPT_HASH_SEPARATOR = "\n"


def prompt_hash(model: str, prompt: str, schema_name: str) -> str:
    key = PROMPT_HASH_SEPARATOR.join((model, prompt, schema_name))
    return hashlib.sha256(key.encode()).hexdigest()


def plain_data(obj: BaseModel | Any) -> Any:
    """JSON-ready form of a pydantic model or a dataclass instance; tuples become lists."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json", by_alias=True)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return json.loads(json.dumps(dataclasses.asdict(obj)))
    raise TypeError(f"artifact must be a pydantic model or a dataclass, got {type(obj)!r}")


def append_json_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
