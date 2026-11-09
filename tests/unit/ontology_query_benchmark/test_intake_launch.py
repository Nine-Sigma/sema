"""The intake launcher pins SEMA's model and embedding routing to the profile and skips .env."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import click
import pytest

from showcase.ontology_query_benchmark import intake_launch
from showcase.ontology_query_benchmark.profile import Profile
from tests.unit.ontology_query_benchmark.conftest import PROFILE_PATH

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
FOREIGN_HOST = "https://foreign.example.test"


@pytest.fixture(autouse=True)
def bare_environ() -> Iterator[None]:
    saved = os.environ.copy()
    os.environ.clear()
    os.environ.update(INTAKE_MODEL_MAX_CALLS="0", INTAKE_MODEL_MAX_COST="0")
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _recording_cli(seen: list[dict[str, Any]]) -> click.Command:
    @click.command(context_settings={"ignore_unknown_options": True})
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    def stub(args: tuple[str, ...]) -> None:
        from sema.cli_intake_utils import intake_model_config
        from sema.models.config import EmbeddingConfig

        model = intake_model_config()
        seen.append({
            "args": args,
            "model": (model.provider, model.model),
            "embedding": (os.environ.get("EMBEDDING_PROVIDER"), EmbeddingConfig().model),
            "databricks": (os.environ.get("DATABRICKS_HOST"),
                           os.environ.get("DATABRICKS_CONFIG_PROFILE")),
        })  # fmt: skip

    return stub


def _launch(cli: click.Command | None, *sema_args: str) -> int | str | None:
    with pytest.raises(SystemExit) as exited:
        intake_launch.main(["--profile", str(PROFILE_PATH), *sema_args], modules=(), cli=cli)
    return exited.value.code


def test_the_cli_sees_the_profile_routing(profile: Profile) -> None:
    seen: list[dict[str, Any]] = []

    code = _launch(_recording_cli(seen), "target", "discover", "src.duckdb", "--tier", "x")

    routing = profile.routing
    assert code == 0
    assert seen == [{
        "args": ("target", "discover", "src.duckdb", "--tier", "x"),
        "model": ("databricks", routing.chat_endpoint),
        "embedding": ("databricks", routing.embedding_endpoint),
        "databricks": (routing.host, routing.databricks_profile),
    }]  # fmt: skip


def test_an_explicit_empty_embedding_provider_stays_off(profile: Profile) -> None:
    os.environ["EMBEDDING_PROVIDER"] = ""
    seen: list[dict[str, Any]] = []

    assert _launch(_recording_cli(seen)) == 0

    assert seen[0]["embedding"] == ("", profile.routing.embedding_endpoint)


@pytest.mark.parametrize(
    ("name", "value"),
    [("OPENROUTER_API_KEY", "key"), ("DATABRICKS_HOST", FOREIGN_HOST),
     ("INTAKE_MODEL_PROVIDER", "openrouter")],
)  # fmt: skip
def test_a_foreign_route_refuses_before_the_cli_runs(name: str, value: str) -> None:
    os.environ[name] = value
    seen: list[dict[str, Any]] = []

    code = _launch(_recording_cli(seen))

    assert seen == []
    assert isinstance(code, str) and name in code


def test_a_dotenv_file_is_not_applied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dotenv.main

    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        f"DATABRICKS_HOST={FOREIGN_HOST}\nEMBEDDING_BASE_URL={FOREIGN_HOST}\n", encoding="utf-8"
    )
    monkeypatch.setattr(dotenv.main, "find_dotenv", lambda *a, **k: str(dotenv_file))
    monkeypatch.chdir(tmp_path)

    assert _launch(None, "target", "--help") == 0

    assert "EMBEDDING_BASE_URL" not in os.environ
    assert os.environ["DATABRICKS_HOST"] != FOREIGN_HOST


def test_the_module_entry_checks_a_fresh_process(tmp_path: Path) -> None:
    command = [sys.executable, "-m", "showcase.ontology_query_benchmark.intake_launch",
               "--profile", str(PROFILE_PATH), "target", "--help"]  # fmt: skip
    clean = {"PATH": os.defpath, "HOME": str(tmp_path)}

    ran = subprocess.run(command, cwd=REPO, env=clean, capture_output=True, text=True)
    refused = subprocess.run(command, cwd=REPO, env={**clean, "OPENROUTER_API_KEY": "key"},
                             capture_output=True, text=True)  # fmt: skip

    assert ran.returncode == 0, ran.stderr
    assert "discover" in ran.stdout
    assert refused.returncode != 0
    assert "OPENROUTER_API_KEY" in refused.stderr and refused.stdout == ""
