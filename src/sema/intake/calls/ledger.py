"""The run ledger: one directory per run holding everything a reviewer or a replay needs.

Layout under ``<root>/<run_id>/``: ``events.jsonl`` and ``sql.jsonl`` (append only),
``calls/<prompt hash>.json`` (one per model call), ``artifacts/<name>.json``.
The ledger never touches the registry; the caller counts a call there on a replay miss.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from sema.intake.calls.ledger_utils import append_json_line, plain_data, prompt_hash, write_json

Clock = Callable[[], datetime]

EVENTS_FILE = "events.jsonl"
SQL_FILE = "sql.jsonl"
CALLS_DIR = "calls"
ARTIFACTS_DIR = "artifacts"
REPLAY_KIND = "replay"
PROVENANCE_KIND = "provenance"


def _utc_now() -> datetime:
    return datetime.now(UTC)


class RunLedger:
    def __init__(self, directory: Path, clock: Clock = _utc_now) -> None:
        self.directory = directory
        self._clock = clock

    def event(self, stage: str, kind: str, payload: dict[str, Any]) -> None:
        append_json_line(
            self.directory / EVENTS_FILE,
            {
                "stage": stage,
                "kind": kind,
                "timestamp": self._clock().isoformat(),
                "payload": payload,
            },
        )

    def sql(
        self,
        stage: str,
        provider: str,
        statement: str,
        value_bearing: bool,
        schema_relation: bool,
    ) -> None:
        append_json_line(
            self.directory / SQL_FILE,
            {
                "stage": stage,
                "provider": provider,
                "statement": statement,
                "value_bearing": value_bearing,
                "schema_relation": schema_relation,
                "timestamp": self._clock().isoformat(),
            },
        )

    def artifact(self, name: str, obj: BaseModel | Any) -> Path:
        path = self.directory / ARTIFACTS_DIR / f"{name}.json"
        write_json(path, plain_data(obj))
        return path

    def record_call(
        self, model: str, prompt: str, schema_name: str, output: dict[str, Any], tokens: int
    ) -> str:
        key = prompt_hash(model, prompt, schema_name)
        write_json(
            self._call_path(key),
            {
                "model": model,
                "prompt": prompt,
                "schema_name": schema_name,
                "output": output,
                "tokens": tokens,
            },
        )
        return key

    def replay(
        self, model: str, prompt: str, schema_name: str, *, stage: str = REPLAY_KIND
    ) -> dict[str, Any] | None:
        key = prompt_hash(model, prompt, schema_name)
        path = self._call_path(key)
        if not path.is_file():
            return None
        recorded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))["output"]
        payload = {"prompt_hash": key, "model": model, "schema_name": schema_name}
        self.event(stage, REPLAY_KIND, payload)
        return recorded

    def provenance(
        self,
        stage: str,
        *,
        model: str,
        prompt_hash: str,
        input_locator: str,
        output_locator: str,
    ) -> None:
        self.event(
            stage,
            PROVENANCE_KIND,
            {
                "model": model,
                "prompt_hash": prompt_hash,
                "input_locator": input_locator,
                "output_locator": output_locator,
            },
        )

    def _call_path(self, key: str) -> Path:
        return self.directory / CALLS_DIR / f"{key}.json"


class Ledger:
    def __init__(self, root: Path, clock: Clock = _utc_now) -> None:
        self.root = root
        self._clock = clock

    def run(self, run_id: str) -> RunLedger:
        directory = self.root / run_id
        directory.mkdir(parents=True, exist_ok=True)
        return RunLedger(directory, self._clock)
