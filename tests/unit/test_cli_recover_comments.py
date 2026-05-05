from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from sema.cli import cli
from sema.ingest.comment_recovery import (
    LiveTableComments,
    ParsedTableComments,
)

pytestmark = pytest.mark.unit


def _stub_dependencies(
    *,
    registered_study: str | None = "study_x",
    target_schema: str = "cbioportal_x",
    parsed: dict[str, ParsedTableComments] | None = None,
    live: dict[str, LiveTableComments] | None = None,
    cache_exists: bool = True,
) -> dict[str, MagicMock]:
    parsed = parsed if parsed is not None else {
        "patient": ParsedTableComments(
            table_comment="cBioPortal clinical patient",
            column_comments={"PATIENT_ID": "Identifier."},
        ),
    }
    live = live if live is not None else {
        "patient": LiveTableComments(
            table_comment=None, column_comments={"PATIENT_ID": ""},
        ),
    }
    extract_mock = MagicMock(return_value=parsed)
    read_mock = MagicMock(return_value=live)
    executor_mock = MagicMock()
    bridge_mock = MagicMock()
    bridge_mock._execute = executor_mock  # not directly used; see patches below
    return {
        "extract": extract_mock, "read": read_mock,
        "executor": executor_mock, "bridge": bridge_mock,
        "registered_study": registered_study,
        "target_schema": target_schema,
        "cache_exists": cache_exists,
    }


def _registry_mock(stubs: dict[str, MagicMock]) -> MagicMock:
    r = MagicMock()
    if stubs["registered_study"] is not None:
        r.find_schema_for_study.return_value = stubs["target_schema"]
    else:
        r.find_schema_for_study.return_value = None
    return r


def _invoke(
    args: list[str], stubs: dict[str, MagicMock], tmp_path: Path,
) -> object:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    if stubs["cache_exists"] and stubs["registered_study"] is not None:
        (cache_root / stubs["registered_study"]).mkdir()
    runner = CliRunner()
    with patch("sema.cli_ingest.Staging") as staging_cls, patch(
        "sema.cli_ingest.StudyRegistry", return_value=_registry_mock(stubs),
    ), patch(
        "sema.cli_ingest._extract_study_comments_lazy",
        stubs["extract"],
    ), patch(
        "sema.cli_ingest.read_databricks_comments", stubs["read"],
    ), patch(
        "sema.cli_ingest._open_recovery_executor",
        return_value=(stubs["executor"], MagicMock()),
    ):
        staging_cls.return_value = MagicMock()
        return runner.invoke(
            cli,
            [
                "ingest", "recover-comments",
                "--duckdb-path", str(tmp_path / "stg.duckdb"),
                "--cache-dir", str(cache_root),
                *args,
            ],
        )


def test_registered_study_runs_with_stubbed_executor(tmp_path: Path) -> None:
    stubs = _stub_dependencies()
    result = _invoke(["--study", "study_x"], stubs, tmp_path)
    assert result.exit_code == 0, result.output
    stubs["executor"].assert_called()
    assert "Columns updated: 1" in result.output
    assert "Table comments updated: 1" in result.output


def test_unregistered_study_exits_nonzero_with_message(tmp_path: Path) -> None:
    stubs = _stub_dependencies(registered_study=None)
    result = _invoke(["--study", "ghost"], stubs, tmp_path)
    assert result.exit_code != 0
    assert "_sema_study_registry" in result.output


def test_dry_run_does_not_execute(tmp_path: Path) -> None:
    stubs = _stub_dependencies()
    result = _invoke(["--study", "study_x", "--dry-run"], stubs, tmp_path)
    assert result.exit_code == 0, result.output
    stubs["executor"].assert_not_called()


def test_force_flag_is_threaded_through(tmp_path: Path) -> None:
    stubs = _stub_dependencies(
        live={
            "patient": LiveTableComments(
                table_comment="existing",
                column_comments={"PATIENT_ID": "existing"},
            ),
        },
    )
    result = _invoke(["--study", "study_x", "--force"], stubs, tmp_path)
    assert result.exit_code == 0, result.output
    stubs["executor"].assert_called()


def test_json_output_is_parseable(tmp_path: Path) -> None:
    stubs = _stub_dependencies()
    result = _invoke(["--study", "study_x", "--json"], stubs, tmp_path)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["study_id"] == "study_x"
    assert payload["target_schema"] == "cbioportal_x"
    assert payload["columns_updated"] >= 1


def test_explicit_overrides_bypass_registry(tmp_path: Path) -> None:
    stubs = _stub_dependencies(registered_study=None)
    cache = tmp_path / "alt_cache"
    cache.mkdir()
    runner = CliRunner()
    with patch("sema.cli_ingest.Staging") as staging_cls, patch(
        "sema.cli_ingest.StudyRegistry", return_value=_registry_mock(stubs),
    ), patch(
        "sema.cli_ingest._extract_study_comments_lazy", stubs["extract"],
    ), patch(
        "sema.cli_ingest.read_databricks_comments", stubs["read"],
    ), patch(
        "sema.cli_ingest._open_recovery_executor",
        return_value=(stubs["executor"], MagicMock()),
    ):
        staging_cls.return_value = MagicMock()
        result = runner.invoke(
            cli,
            [
                "ingest", "recover-comments",
                "--source-cache", str(cache),
                "--target-catalog", "workspace",
                "--target-schema", "cbioportal_x",
                "--duckdb-path", str(tmp_path / "stg.duckdb"),
            ],
        )
    assert result.exit_code == 0, result.output
    stubs["executor"].assert_called()
