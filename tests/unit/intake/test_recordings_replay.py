"""Guard: the committed recordings of sources the role-entry fixes must not touch replay.

F1 (one field per repeated role), F2 (claims in force never offered), and F4 (section
entries asked) change a prompt or an output model only for a relation with a repeated
role, an offered claimed column, or an open section. The diagnosis measured none in the
recorded a2, a7_csv, or a10 runs, so each replays to Review with zero provider calls.
A1 holds all three and is re-recorded after the fix, so it is not listed.
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

# a10 is re-recorded after the described-data rule, which removes its structure call.
UNTOUCHED = ("a2", "a7_csv")


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
