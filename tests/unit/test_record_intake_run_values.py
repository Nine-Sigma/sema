"""The recording script's explicit value approval (task 14.3, Dean's decision 2026-10-21).

Contract added to ``test_record_intake_run.py``'s: ``record_source`` and ``verify_replay``
take ``approver: str | None = None``. With an approver, a run stopped at ``value_ask`` is
answered through ``sema.intake.run.graph.approve_values`` (at most ``MAX_VALUE_ASKS`` times),
each approval is logged in the run ledger as a ``VALUE_APPROVAL_KIND`` event at stage
``value_ask`` carrying the approver, and the run continues to Review; ``recording.json``
carries ``approver`` and ``value_asks`` (approvals made). Without an approver a value ask
still raises ``RecordingError``. ``main`` takes ``--approve-values`` with ``--approver NAME``
(required with the flag) and passes the approver to both steps.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.intake.calls.registry import Registry
from tests.unit.intake.graph_env import (
    ACRIS_DESCRIPTOR,
    REGISTRY_FILE,
    code_tables,
    graph,
    routed_client,
)
from tests.unit.intake.model_calls_env import intake_config
from tests.unit.record_intake_run_env import (
    APPROVER,
    VALUE_SOURCE,
    asking_client,
    model_env,
    stuck_at_values,
    value_source,
)

pytestmark = pytest.mark.unit

REVIEW_PARTS = ("ambiguities", "proposals", "spec", "dry_run")


def utils() -> ModuleType:
    return importlib.import_module("scripts.record_intake_run_utils")


def script() -> ModuleType:
    return importlib.import_module("scripts.record_intake_run")


def record(root: Path, client: Any, approver: str | None, name: str = "work") -> Any:
    return script().record_source(
        value_source(root), recordings=root / "recordings", workdir=root / name,
        config=intake_config(max_calls=200), client=client, approver=approver,
    )


def approvals(ledger_dir: Path) -> list[dict[str, Any]]:
    lines = (ledger_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    return [e for e in events if e["kind"] == utils().VALUE_APPROVAL_KIND]


@pytest.fixture
def approved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: list[str] = []
    real = graph().approve_values

    def spy(run_id: str, **kwargs: Any) -> Any:
        calls.append(run_id)
        return real(run_id, **kwargs)

    monkeypatch.setattr(graph(), "approve_values", spy)
    rec = record(tmp_path, asking_client(), APPROVER)
    return {"root": tmp_path, "rec": rec, "approve_calls": calls}


def test_a_value_ask_without_an_approver_is_still_refused(tmp_path: Path) -> None:
    with pytest.raises(utils().RecordingError, match="value_ask"):
        record(tmp_path, asking_client(), None)
    assert not (tmp_path / "recordings").exists()


def test_an_approver_answers_the_value_ask_and_the_run_reaches_review(
    approved: dict[str, Any],
) -> None:
    rec = approved["rec"]

    assert rec.outcome.node == "review"
    assert approved["approve_calls"] == [rec.run_id]
    assert rec.directory == approved["root"] / "recordings" / VALUE_SOURCE


def test_each_value_approval_is_logged_with_its_approver(approved: dict[str, Any]) -> None:
    rec = approved["rec"]

    (event,) = approvals(approved["root"] / "work" / rec.run_id)
    assert event["stage"] == "value_ask"
    assert event["payload"]["approver"] == APPROVER


def test_the_manifest_records_the_approver_and_the_value_asks(
    approved: dict[str, Any],
) -> None:
    rec = approved["rec"]
    manifest = json.loads((rec.directory / "recording.json").read_text(encoding="utf-8"))

    assert manifest == rec.manifest
    assert (manifest["approver"], manifest["value_asks"]) == (APPROVER, 1)


def test_a_recording_without_values_states_no_approver(tmp_path: Path) -> None:
    source = utils().RecordedSource(
        name="code_tables", path=code_tables(tmp_path), descriptor=ACRIS_DESCRIPTOR
    )
    rec = script().record_source(
        source, recordings=tmp_path / "recordings", workdir=tmp_path / "work",
        config=intake_config(max_calls=200), client=routed_client(), approver=APPROVER,
    )

    assert (rec.manifest["approver"], rec.manifest["value_asks"]) == (None, 0)


def test_replay_follows_the_approved_value_path_with_zero_provider_calls(
    approved: dict[str, Any],
) -> None:
    root, rec = approved["root"], approved["rec"]
    client = routed_client()

    outcome = script().verify_replay(
        value_source(root), recordings=root / "recordings", workdir=root / "replay",
        config=intake_config(max_calls=200), client=client, approver=APPROVER,
    )
    row = Registry(root / "replay" / REGISTRY_FILE).get(outcome.run_id)

    assert client.invoke.call_count == 0
    assert outcome.node == "review" and outcome.run_id != rec.run_id
    assert (row.model_calls, row.tokens, row.cost) == (0, 0, 0.0)
    for part in REVIEW_PARTS:
        assert outcome.payload[part] == rec.outcome.payload[part], part
    assert len(approvals(root / "replay" / outcome.run_id)) == 1


def test_replay_of_a_value_recording_without_an_approver_is_refused(
    approved: dict[str, Any],
) -> None:
    root = approved["root"]

    with pytest.raises(utils().ReplayMismatchError, match=VALUE_SOURCE):
        script().verify_replay(
            value_source(root), recordings=root / "recordings", workdir=root / "replay",
            config=intake_config(max_calls=200), client=routed_client(),
        )


def test_value_asks_past_the_cap_stop_the_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph(), "discover", stuck_at_values())
    monkeypatch.setattr(graph(), "approve_values", stuck_at_values())

    with pytest.raises(utils().RecordingError, match="value_ask"):
        record(tmp_path, MagicMock(), APPROVER)
    assert not (tmp_path / "recordings").exists()


def test_main_requires_an_approver_with_approve_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model_env(monkeypatch, tmp_path, "openrouter")
    monkeypatch.setattr(script(), "record_source", MagicMock(side_effect=AssertionError))

    code = script().main(["--source", "a2", "--approve-values"])

    assert code != 0
    assert "--approver" in capsys.readouterr().err


@pytest.mark.parametrize(("flags", "expected"), [
    (["--approve-values", "--approver", APPROVER], APPROVER),
    ([], None),
])
def test_main_passes_the_approver_to_record_and_replay(
    flags: list[str], expected: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_env(monkeypatch, tmp_path, "openrouter")
    seen: list[Any] = []

    def _step(result: MagicMock) -> Any:
        def _run(_source: Any, **kwargs: Any) -> MagicMock:
            seen.append(kwargs)
            return result

        return _run

    monkeypatch.setattr(script(), "record_source", _step(MagicMock(run_id="r")))
    monkeypatch.setattr(script(), "verify_replay", _step(MagicMock(node="review")))

    code = script().main(["--source", "a2", "--workdir", str(tmp_path / "w"), *flags])

    assert code == 0
    assert [kw["approver"] for kw in seen] == [expected, expected]
