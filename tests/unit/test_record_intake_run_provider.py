"""The recording script's provider guard: OpenRouter only, refused before any run."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.intake.calls.model_calls import IntakeProviderNotSetError
from tests.unit.intake.graph_env import graph
from tests.unit.intake.model_calls_env import intake_config
from tests.unit.record_intake_run_env import ENV_TEXT, forbid, model_env

pytestmark = pytest.mark.unit


def utils() -> ModuleType:
    return importlib.import_module("scripts.record_intake_run_utils")


def script() -> ModuleType:
    return importlib.import_module("scripts.record_intake_run")


def sources() -> dict[str, Any]:
    return {source.name: source for source in script().RECORDED_SOURCES}


# --- the provider guard -----------------------------------------------------------------


def test_check_provider_requires_an_explicit_provider() -> None:
    with pytest.raises(IntakeProviderNotSetError):
        utils().check_provider(intake_config(provider=None))
    utils().check_provider(intake_config(provider="openrouter"))
    assert utils().RECORDING_PROVIDER == "openrouter"


def test_check_provider_refuses_any_other_provider_by_name() -> None:
    with pytest.raises(utils().ProviderRefusedError, match="databricks"):
        utils().check_provider(intake_config(provider="databricks"))
    assert issubclass(utils().ProviderRefusedError, ValueError)


def test_record_source_refuses_another_provider_before_any_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph(), "discover", forbid("discover"))
    client = MagicMock()

    with pytest.raises(utils().ProviderRefusedError, match="databricks"):
        script().record_source(
            sources()["a2"], recordings=tmp_path / "recordings", workdir=tmp_path / "work",
            config=intake_config(provider="databricks"), client=client,
        )
    assert client.invoke.call_count == 0
    assert not (tmp_path / "recordings").exists()


@pytest.mark.parametrize(("provider", "named"), [(None, "INTAKE_MODEL_PROVIDER"),
                                                 ("databricks", "databricks")])
def test_main_refuses_without_openrouter_and_leaves_env_alone(
    provider: str | None, named: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = model_env(monkeypatch, tmp_path, provider)
    monkeypatch.setattr(script(), "record_source", forbid("record_source"))

    code = script().main(["--source", "a2", "--recordings", str(tmp_path / "rec")])

    assert code != 0
    assert named in capsys.readouterr().err
    assert env.read_text(encoding="utf-8") == ENV_TEXT


def test_main_records_then_verifies_each_named_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = model_env(monkeypatch, tmp_path, "openrouter")
    steps: list[tuple[str, str, Any]] = []

    def _record(source: Any, **kwargs: Any) -> Any:
        steps.append(("record", source.name, kwargs))
        return MagicMock(run_id="r")

    def _verify(source: Any, **kwargs: Any) -> Any:
        steps.append(("verify", source.name, kwargs))
        return MagicMock(node="review")

    monkeypatch.setattr(script(), "record_source", _record)
    monkeypatch.setattr(script(), "verify_replay", _verify)
    recordings = tmp_path / "rec"

    code = script().main(["--source", "a2", "--source", "a3", "--recordings", str(recordings),
                          "--workdir", str(tmp_path / "work")])

    assert code == 0
    assert [(step, name) for step, name, _ in steps] == [
        ("record", "a2"), ("verify", "a2"), ("record", "a3"), ("verify", "a3"),
    ]
    for _, _, kwargs in steps:
        assert kwargs["recordings"] == recordings
        assert kwargs["config"].provider == "openrouter"
    assert env.read_text(encoding="utf-8") == ENV_TEXT
