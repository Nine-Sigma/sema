"""Guard: committed recordings of known sources replay to Review with zero provider calls.

Each listed source was re-recorded after the generic intake fixes; a prompt or output
model change that alters a recorded call key makes its replay refuse a call and fail.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from sema.models.config import ModelPrice
from tests.unit.intake.model_calls_env import intake_config

pytestmark = pytest.mark.unit

UNTOUCHED = ("a1", "a2", "a7_csv", "a10")


def script() -> ModuleType:
    return importlib.import_module("scripts.record_intake_run")


def utils() -> ModuleType:
    return importlib.import_module("scripts.record_intake_run_utils")


def _offline_client() -> MagicMock:
    client = MagicMock()
    client.invoke.side_effect = AssertionError("a recorded call missed and reached the client")
    return client


@pytest.mark.parametrize("name", UNTOUCHED)
def test_committed_recording_replays_with_zero_provider_calls(name: str, tmp_path: Path) -> None:
    recordings: Path = utils().RECORDINGS_DIR
    manifest = json.loads((recordings / name / "recording.json").read_text())
    model = manifest["model"]
    config = intake_config(
        model=model,
        max_calls=200,
        prices={model: ModelPrice(input_per_1k=0.0001, output_per_1k=0.0005)},
    )
    (source,) = [s for s in script().RECORDED_SOURCES if s.name == name]

    outcome = script().verify_replay(
        source,
        recordings=recordings,
        workdir=tmp_path,
        config=config,
        client=_offline_client(),
        approver=manifest.get("approver"),
    )

    assert outcome.node == "review"
