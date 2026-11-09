"""The benchmark's discovery input is role-neutral, and its env form can skip ``.env``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from sema.intake.eligibility_utils import RunDescriptor

pytestmark = pytest.mark.unit

CONFIG = Path(__file__).parents[3] / "showcase" / "ontology_query_benchmark" / "config"
DESCRIPTOR = CONFIG / "intake_standard_library.run.yaml"
SELECTED = ("standard_library_data_element", "standard_library_concept")
LOCATION_KEYS = {"id", "identity_system", "release", "relations"}
ROLE_MAP_COLUMNS = (
    "data_element_prt_id", "data_element_title", "data_element_definition",
    "permissible_value_prt_id", "permissible_value_title", "permissible_value_concept_code",
    "concept_prt_id", "concept_title", "prt_preferred_label", "concept_code", "vocabulary_id",
    "concept_athena_id", "definition", "phi",
)
SENTINEL = "SEMA_BENCH_DOTENV_SENTINEL"


def test_descriptor_states_locations_only() -> None:
    raw = yaml.safe_load(DESCRIPTOR.read_text(encoding="utf-8"))

    descriptor = RunDescriptor.model_validate(raw)

    assert set(raw) == LOCATION_KEYS
    assert descriptor.relations == SELECTED


def test_descriptor_names_no_role_map_column() -> None:
    values = yaml.safe_load(DESCRIPTOR.read_text(encoding="utf-8")).values()
    stated = {str(v).lower() for v in values if not isinstance(v, list)}

    assert not stated & set(ROLE_MAP_COLUMNS)


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
