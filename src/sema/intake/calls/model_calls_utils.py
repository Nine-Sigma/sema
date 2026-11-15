"""Helpers for the intake model-call path: recordings, pricing, and event payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from sema.intake.calls.ledger import CALLS_DIR
from sema.llm_client import InvocationStats, LLMStageError
from sema.llm_client_utils import approx_tokens
from sema.models.config import ModelPrice

TOKENS_PER_PRICE_UNIT = 1000
RECORDING_MANIFEST = "recording.json"
AGREEMENT_FIELD = "agreement"
EMBEDDINGS_FIELD = "embeddings"


def read_recording(recordings_dir: Path | None, key: str) -> dict[str, Any] | None:
    """The call record for `key` in a read-only recordings directory, or None."""
    if recordings_dir is None:
        return None
    path = recordings_dir / CALLS_DIR / f"{key}.json"
    if not path.is_file():
        return None
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return record


def _recorded_flag(recordings_dir: Path, field: str) -> bool:
    path = recordings_dir / RECORDING_MANIFEST
    if not path.is_file():
        return False
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return bool(manifest.get(field, False))


def recorded_agreement(recordings_dir: Path) -> bool:
    """The agreement flag a recording was made with; a manifest without it, or no
    manifest, predates agreement and replays with one ask."""
    return _recorded_flag(recordings_dir, AGREEMENT_FIELD)


def recorded_embeddings(recordings_dir: Path) -> bool:
    """Whether a recording was made with an embedder; without the flag it was not, so a
    replay builds none."""
    return _recorded_flag(recordings_dir, EMBEDDINGS_FIELD)


def total_tokens(stats: InvocationStats) -> int:
    return stats.prompt_tokens + stats.completion_tokens


def call_cost(stats: InvocationStats, price: ModelPrice) -> float:
    return (
        stats.prompt_tokens / TOKENS_PER_PRICE_UNIT * price.input_per_1k
        + stats.completion_tokens / TOKENS_PER_PRICE_UNIT * price.output_per_1k
    )


def prompt_tokens(prompt: str, output_model: type[BaseModel]) -> int:
    """What reaches the provider: the prompt and its output schema, at the client's estimate."""
    schema = json.dumps(output_model.model_json_schema())
    return approx_tokens(len(prompt) + len(schema))


def call_payload(key: str, model: str, schema_name: str) -> dict[str, Any]:
    return {"prompt_hash": key, "model": model, "schema_name": schema_name}


def step_errors(error: LLMStageError) -> list[dict[str, str]]:
    return [{"step": step, "error": str(cause)} for step, cause in error.step_errors]
