"""`sema target discover` hands the graph SEMA's embedder only when one is configured.

Pinned (graph follow-up adds ``discover(..., embedder=None)``): with ``EMBEDDING_PROVIDER`` in
the environment, discover passes ``embedder=cli_factories._get_embedder(EmbeddingConfig())``,
resolved through ``sema.cli_factories`` at call time; without it, ``embedder=None`` and the
factory is never called. The real graph runs with the embedder dropped, so no provider is hit.

Replay (design decision 4, bug-832): under ``INTAKE_MODEL_MAX_CALLS=0`` with a recordings
directory, neither ``discover`` nor ``approve`` builds an embedder whatever
``EMBEDDING_PROVIDER`` says, and one log line names the variable and the zero cap; a recording
whose ``recording.json`` states ``"embeddings": true`` replays with the embedder.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from sema.models.config import EmbeddingConfig
from tests.unit.cli_intake_env import (
    Cli,
    captured_logs,
    intake_cli,
    ok_or_raise,
    spy_discover,
    zero_cap_replay,
)
from tests.unit.intake.graph_env import A2_DESCRIPTOR
from tests.unit.intake.structure_env import A2_WORKBOOK

pytestmark = pytest.mark.unit

FACTORY = "sema.cli_factories._get_embedder"
EMBEDDING_ENV = {"EMBEDDING_PROVIDER": "openrouter", "EMBEDDING_MODEL": "vendor-a/embed-x"}
STRAY_EMBEDDING_ENV = {
    "EMBEDDING_PROVIDER": "not-a-provider",
    "EMBEDDING_MODEL": "vendor-a/embed-x",
}
ZERO_CAP = re.compile(r"\b0\b|zero", re.IGNORECASE)


@pytest.mark.parametrize("configured", [True, False], ids=["provider-set", "provider-unset"])
def test_discover_passes_the_configured_embedder_or_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured: bool
) -> None:
    cli: Cli = intake_cli(tmp_path / "runs", monkeypatch)
    for name, value in EMBEDDING_ENV.items():
        if configured:
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)
    built: list[EmbeddingConfig] = []
    embedder = object()

    def _factory(config: EmbeddingConfig) -> Any:
        built.append(config)
        return embedder

    monkeypatch.setattr(FACTORY, _factory)
    seen = spy_discover(monkeypatch, drop=("embedder",))

    cli.discover(A2_WORKBOOK, A2_DESCRIPTOR)

    ((_, kwargs),) = seen
    assert "embedder" in kwargs
    if configured:
        assert kwargs["embedder"] is embedder
        (config,) = built
        assert (config.provider, config.model) == tuple(EMBEDDING_ENV.values())
    else:
        assert kwargs["embedder"] is None
        assert built == []


def _refuse_embedder(config: EmbeddingConfig) -> Any:
    raise AssertionError("embedder built under a zero cap")


def _stray_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in STRAY_EMBEDDING_ENV.items():
        monkeypatch.setenv(name, value)


def _provider_lines(messages: list[str]) -> list[str]:
    return [message for message in messages if "EMBEDDING_PROVIDER" in message]


def _discover(cli: Cli) -> Any:
    return ok_or_raise(cli.run("discover", A2_WORKBOOK, "--descriptor", A2_DESCRIPTOR))


def _zero_cap_discover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    cli = intake_cli(tmp_path / "runs", monkeypatch)
    zero_cap_replay(monkeypatch, tmp_path / "recordings", max_calls=0)
    _stray_provider(monkeypatch)
    monkeypatch.setattr(FACTORY, _refuse_embedder)
    seen = spy_discover(monkeypatch, drop=("embedder",))
    with captured_logs() as messages:
        _discover(cli)
    return seen, messages


def test_discover_builds_no_embedder_under_a_zero_cap_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen, _ = _zero_cap_discover(tmp_path, monkeypatch)

    ((_, kwargs),) = seen
    assert "embedder" in kwargs
    assert kwargs["embedder"] is None


def test_discover_logs_once_that_the_zero_cap_skips_the_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, messages = _zero_cap_discover(tmp_path, monkeypatch)

    (line,) = _provider_lines(messages)
    assert ZERO_CAP.search(line), line


def test_discover_builds_the_embedder_when_the_cap_is_above_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = intake_cli(tmp_path / "runs", monkeypatch)
    zero_cap_replay(monkeypatch, tmp_path / "recordings", max_calls=1)
    _stray_provider(monkeypatch)
    built: list[EmbeddingConfig] = []
    embedder = object()

    def _factory(config: EmbeddingConfig) -> Any:
        built.append(config)
        return embedder

    monkeypatch.setattr(FACTORY, _factory)
    seen = spy_discover(monkeypatch, drop=("embedder",))
    with captured_logs() as messages:
        _discover(cli)

    ((_, kwargs),) = seen
    assert kwargs["embedder"] is embedder
    (config,) = built
    assert (config.provider, config.model) == tuple(STRAY_EMBEDDING_ENV.values())
    assert _provider_lines(messages) == []


def test_approve_builds_no_embedder_under_a_zero_cap_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = intake_cli(tmp_path / "runs", monkeypatch)
    for name in STRAY_EMBEDDING_ENV:
        monkeypatch.delenv(name, raising=False)
    run = cli.reviewed(A2_WORKBOOK, A2_DESCRIPTOR)
    zero_cap_replay(monkeypatch, tmp_path / "recordings", max_calls=0)
    _stray_provider(monkeypatch)
    monkeypatch.setattr(FACTORY, _refuse_embedder)

    printed = ok_or_raise(cli.run("approve", run.run_id, "--approver", "dean", "--reject"))

    assert printed["status"] == "rejected"


def test_discover_builds_the_embedder_for_a_replay_recorded_with_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = intake_cli(tmp_path / "runs", monkeypatch)
    recordings = tmp_path / "recordings"
    zero_cap_replay(monkeypatch, recordings, max_calls=0)
    (recordings / "recording.json").write_text('{"agreement": true, "embeddings": true}\n')
    _stray_provider(monkeypatch)
    built: list[EmbeddingConfig] = []
    embedder = object()

    def _factory(config: EmbeddingConfig) -> Any:
        built.append(config)
        return embedder

    monkeypatch.setattr(FACTORY, _factory)
    seen = spy_discover(monkeypatch, drop=("embedder",))
    with captured_logs() as messages:
        _discover(cli)

    ((_, kwargs),) = seen
    assert kwargs["embedder"] is embedder
    (config,) = built
    assert (config.provider, config.model) == tuple(STRAY_EMBEDDING_ENV.values())
    assert _provider_lines(messages) == []
