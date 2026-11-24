"""Done-when 5 (N6): an approved intake run as the build's input.

`BuildConfig.intake_run` names an intake run folder that `write_intake_run` leaves as intake
does: a registry row beside the folder, the spec as bare JSON and the staging file. Only a run
at stage `import` with status `approved` resolves; it sets `concept_spec` and
`concept_staging`, and every other case is a config error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sema.models.assertions import Assertion
from sema.models.config import BuildConfig, DuckDBConfig
from sema.pipeline.coded_columns import run_coded_columns
from tests.support.connector_fixture_declared import prepare_declared_step, write_intake_run
from tests.support.connector_fixture_identity import IDENTITY_CATALOG, IDENTITY_SCHEMA
from tests.support.connector_fixture_identity_runs import (
    STEP_RUN_ID,
    StepGraph,
    StepRun,
    fixture_connector,
    run_step,
)

pytestmark = pytest.mark.unit


def _config(**fields: Any) -> BuildConfig:
    assert "intake_run" in BuildConfig.model_fields, "BuildConfig has no intake_run field"
    return BuildConfig(source="duckdb", catalog=IDENTITY_CATALOG, schemas=[IDENTITY_SCHEMA],
                       **fields)


def test_an_approved_run_resolves_the_spec_and_the_staging_file(tmp_path: Path) -> None:
    run = write_intake_run(tmp_path, stage="import", status="approved")

    config = _config(intake_run=str(run))

    assert Path(str(config.concept_spec)) == run / "artifacts" / "spec.json"
    assert Path(str(config.concept_staging)) == run / "staging.duckdb"


@pytest.mark.parametrize(("stage", "status"), [
    ("review", "waiting_review"), ("import", "rejected"), ("review", "approved"),
])
def test_a_run_not_approved_at_import_is_a_config_error(
    tmp_path: Path, stage: str, status: str,
) -> None:
    run = write_intake_run(tmp_path, stage=stage, status=status)

    with pytest.raises(ValidationError):
        _config(intake_run=str(run))


def test_the_row_read_is_the_one_named_by_the_folder(tmp_path: Path) -> None:
    write_intake_run(tmp_path, stage="import", status="approved")
    waiting = write_intake_run(tmp_path, stage="review", status="waiting_review")

    with pytest.raises(ValidationError):
        _config(intake_run=str(waiting))


def test_a_missing_folder_is_a_config_error(tmp_path: Path) -> None:
    write_intake_run(tmp_path, stage="import", status="approved")

    with pytest.raises(ValidationError):
        _config(intake_run=str(tmp_path / "absent-run"))


@pytest.mark.parametrize("part", ["artifacts/spec.json", "staging.duckdb"])
def test_a_missing_spec_or_staging_file_is_a_config_error(tmp_path: Path, part: str) -> None:
    run = write_intake_run(tmp_path, stage="import", status="approved")
    (run / part).rename(run / f"{part}.moved")

    with pytest.raises(ValidationError):
        _config(intake_run=str(run))


@pytest.mark.parametrize("field", ["concept_spec", "concept_staging"])
def test_intake_run_beside_a_concept_file_is_a_config_error(tmp_path: Path, field: str) -> None:
    run = write_intake_run(tmp_path, stage="import", status="approved")

    with pytest.raises(ValidationError):
        _config(intake_run=str(run), **{field: str(run / "staging.duckdb")})


def _written(step: StepRun) -> list[tuple[str, str, str | None, str]]:
    def key(a: Assertion) -> tuple[str, str, str | None, str]:
        return (a.predicate.value, a.subject_ref, a.object_ref,
                json.dumps(a.payload, sort_keys=True, default=str))

    return sorted(key(a) for a in step.built().assertions)


def _coded_step(prepared: StepRun) -> None:
    run_coded_columns(prepared.graph, prepared.connector, prepared.config,  # type: ignore[arg-type]
                      [IDENTITY_SCHEMA], STEP_RUN_ID)


def test_the_build_from_an_approved_run_writes_what_the_two_files_write(tmp_path: Path) -> None:
    (tmp_path / "files").mkdir()
    by_files = run_step(prepare_declared_step(tmp_path / "files"), _coded_step)
    database = Path(by_files.config.duckdb.path)
    run = write_intake_run(tmp_path / "runs", stage="import", status="approved")
    config = _config(duckdb=DuckDBConfig(path=str(database)), skip_embeddings=True,
                     table_workers=1, intake_run=str(run))
    by_run = run_step(StepRun(StepGraph(by_files.graph.columns, []), fixture_connector(database),
                              config, by_files.refs, by_files.dictionary), _coded_step)

    assert _written(by_files)
    assert _written(by_run) == _written(by_files)


def test_a_run_under_a_folder_named_with_a_hash_resolves_and_writes_nothing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs#1"
    run = write_intake_run(root, stage="import", status="approved")

    config = _config(intake_run=str(run))

    assert Path(str(config.concept_spec)) == run / "artifacts" / "spec.json"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["runs#1"]
