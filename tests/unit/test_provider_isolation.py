"""A unit test never sees a provider route, whatever the shell exports or a .env holds."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.unit

PROVIDER_ROUTES = {
    "EMBEDDING_PROVIDER",
    "EMBEDDING_BASE_URL",
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "OPENROUTER_API_KEY",
    "INTAKE_MODEL_PROVIDER",
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_CONFIG_PROFILE",
    "MLFLOW_TRACKING_URI",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_API_KEY",
}


def test_no_provider_route_reaches_a_unit_test() -> None:
    assert sorted(PROVIDER_ROUTES & os.environ.keys()) == []


def test_the_sema_cli_does_not_apply_a_dotenv_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dotenv.main

    from sema.cli import cli

    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("EMBEDDING_PROVIDER=custom\n", encoding="utf-8")
    monkeypatch.setattr(dotenv.main, "find_dotenv", lambda *a, **k: str(dotenv_file))
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)

    assert CliRunner().invoke(cli, ["target", "--help"]).exit_code == 0

    assert "EMBEDDING_PROVIDER" not in os.environ


def test_the_databricks_sdk_reads_no_profile_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import databricks.sdk.config
    from databricks.sdk.core import Config

    profiles = "[DEFAULT]\nhost = https://foreign.example.test\ntoken = fake-token\n"
    (tmp_path / ".databrickscfg").write_text(profiles, encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    reached: list[str] = []

    def _refuse(host: str) -> None:
        reached.append(host)
        raise ConnectionRefusedError(host)

    monkeypatch.setattr(databricks.sdk.config, "get_host_metadata", _refuse)

    with pytest.raises(ValueError, match="default auth"):
        Config()
    assert reached == []
