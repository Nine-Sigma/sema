"""Fakes for the recording script tests: model settings, a stand-in graph, a read guard."""

from __future__ import annotations

import builtins
import io
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sema.intake.ledger import Ledger
from sema.intake.registry import Registry
from tests.unit.intake.graph_env import REGISTRY_FILE, graph
from tests.unit.intake.model_calls_env import MODEL, choice, numbered_prompt
from tests.unit.intake.shapes.shapes_env import FIXTURES

HELD_OUT = ("heldout", "heldout2")
REVIEW_PARTS = ("ambiguities", "proposals", "spec", "dry_run")
ENV_PREFIX = "INTAKE_MODEL_"
ENV_TEXT = "INTAKE_MODEL_PROVIDER=databricks\n"


def forbid(name: str) -> Callable[..., Any]:
    def _refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(f"the recording script called {name}")

    return _refuse


def model_env(monkeypatch: pytest.MonkeyPatch, cwd: Path, provider: str | None) -> Path:
    """The intake model settings in the environment, beside a ``.env`` that must not change."""
    for name in [key for key in os.environ if key.startswith(ENV_PREFIX)]:
        monkeypatch.delenv(name)
    values = {"MODEL": MODEL, "MAX_CALLS": "200", "MAX_COST": "1.0"}
    if provider is not None:
        values["PROVIDER"] = provider
    for key, value in values.items():
        monkeypatch.setenv(ENV_PREFIX + key, value)
    env = cwd / ".env"
    env.write_text(ENV_TEXT, encoding="utf-8")
    monkeypatch.chdir(cwd)
    return env


def fake_discover(node: str = "review") -> Any:
    """Stands in for the graph: one recorded call in the run ledger, then ``node``."""
    seen: list[dict[str, Any]] = []

    def _discover(source: Path, *, root: Path, run_id: str | None = None, **kwargs: Any) -> Any:
        run_id = run_id or "fake-run"
        seen.append({"source": source, **kwargs})
        output = choice()
        Ledger(root).run(run_id).record_call(
            MODEL, numbered_prompt(1), type(output).__name__, output.model_dump(mode="json"), 1500
        )
        Registry(root / REGISTRY_FILE).register(run_id, source=source.name, stage="land")
        payload = {part: None for part in REVIEW_PARTS}
        return graph().RunOutcome(run_id, node, payload)

    _discover.seen = seen  # type: ignore[attr-defined]
    return _discover


def _under_held_out(path: Path) -> bool:
    resolved = path.resolve()
    return any(resolved.is_relative_to((FIXTURES / name).resolve()) for name in HELD_OUT)


def guard_held_out_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test the moment anything opens a file under a held-out directory."""
    original = io.open

    def _open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, Path)) and _under_held_out(Path(file)):
            raise AssertionError(f"held-out file opened: {file}")
        return original(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", _open)
    monkeypatch.setattr(builtins, "open", _open)
