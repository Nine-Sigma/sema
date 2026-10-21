"""Shared fakes for the intake model-call path: a mocked client, configs, and a wired run.

`sema.intake.model_calls` and `IntakeModelConfig` are imported inside the helpers so a
missing module fails each test instead of the whole collection.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock

from pydantic import BaseModel, ConfigDict

from sema.intake.ledger import Ledger, RunLedger
from sema.intake.registry import Registry, RunRecord
from sema.llm_client import InvocationStats

MODEL = "vendor-a/model-x"
STAGE = "structure"
PROMPT_BODY = "Choose the template for entry e1.\nMembers: a, b."
PROMPT = "# prompt: structure v1\n" + PROMPT_BODY
PROMPT_V2 = "# prompt: structure v2\n" + PROMPT_BODY
INPUT_PER_1K = 0.002
OUTPUT_PER_1K = 0.008


class Choice_v1(BaseModel):  # noqa: N801 - the version suffix is part of the replay key
    model_config = ConfigDict(extra="forbid")

    choice: Literal["a", "b", "abstain"]
    confidence: float
    reason: str


class Choice_v2(BaseModel):  # noqa: N801
    model_config = ConfigDict(extra="forbid")

    choice: Literal["a", "b", "abstain"]
    confidence: float
    reason: str


def choice(member: Literal["a", "b", "abstain"] = "a") -> Choice_v1:
    return Choice_v1(choice=member, confidence=0.8, reason="named in the header")


def stats(prompt_tokens: int = 1000, completion_tokens: int = 500) -> InvocationStats:
    return InvocationStats(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def numbered_prompt(index: int) -> str:
    return f"# prompt: structure v1\nChoose the template for entry e{index}."


def fake_client(
    outputs: Iterable[BaseModel | Exception] = (),
    per_call_stats: InvocationStats | None = None,
    served_by: str = "structured_output",
) -> MagicMock:
    """A mocked `LLMClient`: invoke sets `last_stats` and `last_path`, then returns or raises."""
    client = MagicMock()
    client.last_stats = InvocationStats()
    client.last_path = None
    pending = list(outputs)
    call_stats = per_call_stats or stats()

    def _invoke(*_args: Any, **_kwargs: Any) -> BaseModel:
        client.last_stats = call_stats
        client.last_path = None
        outcome = pending.pop(0) if pending else choice()
        if isinstance(outcome, Exception):
            raise outcome
        client.last_path = served_by
        return outcome

    client.invoke.side_effect = _invoke
    return client


def intake_config(**overrides: Any) -> Any:
    from sema.models.config import IntakeModelConfig, ModelPrice

    values: dict[str, Any] = {
        "provider": "openrouter",
        "model": MODEL,
        "max_calls": 10,
        "max_cost": 1.0,
        "prices": {MODEL: ModelPrice(input_per_1k=INPUT_PER_1K, output_per_1k=OUTPUT_PER_1K)},
        "recordings_dir": None,
    }
    values.update(overrides)
    return IntakeModelConfig(**values)


@dataclass
class WiredRun:
    calls: Any
    client: MagicMock
    ledger: RunLedger
    registry: Registry
    run_id: str

    def row(self) -> RunRecord:
        matches = [r for r in self.registry.list() if r.run_id == self.run_id]
        assert len(matches) == 1
        return matches[0]

    def events(self, kind: str) -> list[dict[str, Any]]:
        path = self.ledger.directory / "events.jsonl"
        if not path.is_file():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [e for e in map(json.loads, lines) if e["kind"] == kind]

    def call_files(self) -> list[str]:
        return sorted(p.name for p in (self.ledger.directory / "calls").glob("*.json"))

    def call_record(self, key: str) -> dict[str, Any]:
        path = self.ledger.directory / "calls" / f"{key}.json"
        record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return record


def wired_run(
    root: Path,
    client: MagicMock,
    config: Any | None = None,
    run_id: str | None = None,
) -> WiredRun:
    from sema.intake.model_calls import ModelCalls

    run_id = run_id or str(uuid.uuid4())
    ledger = Ledger(root / "runs").run(run_id)
    registry = Registry(root / "registry.sqlite")
    registry.register(run_id, source="source-under-test", stage=STAGE)
    calls = ModelCalls(client, ledger, registry, run_id, config or intake_config())
    return WiredRun(calls, client, ledger, registry, run_id)


def record_in(directory: Path, prompt: str, output: BaseModel, tokens: int = 1500) -> str:
    """Write a call record with the ledger's own layout; returns its key."""
    return RunLedger(directory).record_call(
        MODEL, prompt, type(output).__name__, output.model_dump(mode="json"), tokens
    )


def files_under(directory: Path) -> list[str]:
    return sorted(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file())

