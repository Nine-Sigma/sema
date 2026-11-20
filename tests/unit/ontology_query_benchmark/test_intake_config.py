"""The intake CLI's env form can skip ``.env``; the profile descriptor is checked elsewhere."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.unit

SENTINEL = "SEMA_BENCH_DOTENV_SENTINEL"


def _root_env_after(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str | None:
    import dotenv.main

    from sema.cli import cli

    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(f"{SENTINEL}=from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(dotenv.main, "find_dotenv", lambda *a, **k: str(dotenv_file))
    monkeypatch.delenv(SENTINEL, raising=False)
    try:
        CliRunner().invoke(cli, ["target", "--help"], catch_exceptions=False)
        return os.environ.get(SENTINEL)
    finally:
        os.environ.pop(SENTINEL, None)


def test_dotenv_disabled_switch_keeps_dotenv_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")

    assert _root_env_after(tmp_path, monkeypatch) is None


def test_without_the_switch_the_cli_loads_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)

    assert _root_env_after(tmp_path, monkeypatch) == "from-dotenv"
