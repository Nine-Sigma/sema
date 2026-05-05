from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from scripts.migrate_cbioportal_to_namespaced import _rename_schema, main
from sema.ingest.comment_recovery import ParsedTableComments
from sema.ingest.duckdb_staging import Staging

pytestmark = pytest.mark.unit


def _staging_with_legacy(tmp_path: Path) -> Staging:
    s = Staging(str(tmp_path / "stg.duckdb"))
    s.execute('CREATE SCHEMA IF NOT EXISTS "cbioportal"')
    s.execute(
        'CREATE TABLE "cbioportal"."patient" '
        '(PATIENT_ID VARCHAR, AGE INTEGER)'
    )
    s.execute("INSERT INTO \"cbioportal\".\"patient\" VALUES ('P-001', 42)")
    return s


def test_rename_schema_reapplies_column_and_table_comments(tmp_path: Path) -> None:
    staging = _staging_with_legacy(tmp_path)

    def comment_source(table: str) -> ParsedTableComments:
        if table == "patient":
            return ParsedTableComments(
                table_comment="patient table comment",
                column_comments={
                    "PATIENT_ID": "Patient identifier.",
                    "AGE": "Age at diagnosis.",
                },
            )
        return ParsedTableComments(table_comment=None, column_comments={})

    _rename_schema(
        staging, "cbioportal", "cbioportal_x", comment_source=comment_source,
    )
    info = staging.describe("cbioportal_x", "patient")
    assert info.columns["PATIENT_ID"].comment == "Patient identifier."
    assert info.columns["AGE"].comment == "Age at diagnosis."
    assert info.table_comment == "patient table comment"
    staging.close()


def test_rename_schema_without_comment_source_completes(tmp_path: Path) -> None:
    staging = _staging_with_legacy(tmp_path)
    _rename_schema(staging, "cbioportal", "cbioportal_x")
    info = staging.describe("cbioportal_x", "patient")
    assert "PATIENT_ID" in info.columns
    staging.close()


def test_rename_schema_warns_when_comment_source_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    staging = _staging_with_legacy(tmp_path)

    def failing_source(table: str) -> ParsedTableComments:
        raise FileNotFoundError(f"no cache for {table}")

    _rename_schema(
        staging, "cbioportal", "cbioportal_x",
        comment_source=failing_source,
    )
    info = staging.describe("cbioportal_x", "patient")
    assert "PATIENT_ID" in info.columns
    assert info.columns["PATIENT_ID"].comment in (None, "")
    staging.close()


def test_migration_idempotent_when_target_already_exists(tmp_path: Path) -> None:
    duckdb_path = tmp_path / "stg.duckdb"
    staging = Staging(str(duckdb_path))
    staging.execute('CREATE SCHEMA IF NOT EXISTS "cbioportal_x"')
    staging.execute(
        'CREATE TABLE "cbioportal_x"."patient" (PATIENT_ID VARCHAR)'
    )
    staging.close()

    runner = CliRunner()
    cache_root = tmp_path / "cache"
    (cache_root / "study_x").mkdir(parents=True)
    with patch.dict("os.environ", {
        "INGEST_DUCKDB_PATH": str(duckdb_path),
        "INGEST_CACHE_DIR": str(cache_root),
    }):
        result = runner.invoke(
            main,
            ["--duckdb-path", str(duckdb_path), "--study-id", "study_x"],
        )
    assert result.exit_code == 0, result.output


def test_migration_with_full_flow_preserves_comments(tmp_path: Path) -> None:
    duckdb_path = tmp_path / "stg.duckdb"
    staging = Staging(str(duckdb_path))
    staging.execute('CREATE SCHEMA IF NOT EXISTS "cbioportal"')
    staging.execute(
        'CREATE TABLE "cbioportal"."patient" (PATIENT_ID VARCHAR)'
    )
    staging.close()

    cache_root = tmp_path / "cache"
    (cache_root / "study_x").mkdir(parents=True)
    (cache_root / "study_x" / "data_clinical_patient.txt").write_text(
        "#Patient Identifier\n"
        "#Identifier to uniquely specify a patient.\n"
        "#STRING\n"
        "#1\n"
        "PATIENT_ID\n"
        "P-001\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    with patch.dict("os.environ", {
        "INGEST_DUCKDB_PATH": str(duckdb_path),
        "INGEST_CACHE_DIR": str(cache_root),
    }):
        result = runner.invoke(
            main,
            ["--duckdb-path", str(duckdb_path), "--study-id", "study_x"],
        )
    assert result.exit_code == 0, result.output

    staging = Staging(str(duckdb_path))
    info = staging.describe("cbioportal_study_x", "patient")
    assert (
        info.columns["PATIENT_ID"].comment
        == "Identifier to uniquely specify a patient."
    )
    staging.close()
