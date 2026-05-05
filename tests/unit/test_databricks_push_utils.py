from __future__ import annotations

import pytest

from sema.ingest.databricks_push_utils import (
    build_alter_column_comment_sql,
    build_alter_table_comment_sql,
)

pytestmark = pytest.mark.unit


def test_build_alter_column_comment_sql_quotes_identifiers() -> None:
    sql = build_alter_column_comment_sql(
        "workspace", "cbioportal_x", "patient", "PATIENT_ID",
        "Identifier to uniquely specify a patient.",
    )
    assert sql == (
        "ALTER TABLE `workspace`.`cbioportal_x`.`patient` "
        "ALTER COLUMN `PATIENT_ID` "
        "COMMENT 'Identifier to uniquely specify a patient.'"
    )


def test_build_alter_column_comment_sql_escapes_single_quotes() -> None:
    sql = build_alter_column_comment_sql(
        "workspace", "cbioportal_x", "patient", "PATIENT_ID",
        "Patient's identifier",
    )
    assert "'Patient''s identifier'" in sql


def test_build_alter_column_comment_sql_empty_comment_clears() -> None:
    sql = build_alter_column_comment_sql(
        "workspace", "cbioportal_x", "patient", "PATIENT_ID", "",
    )
    assert sql.endswith("COMMENT ''")


def test_build_alter_column_comment_sql_rejects_semicolon_in_identifier() -> None:
    with pytest.raises(ValueError):
        build_alter_column_comment_sql(
            "workspace", "x; DROP TABLE y", "patient", "PATIENT_ID", "x",
        )


def test_build_alter_column_comment_sql_rejects_backtick_in_identifier() -> None:
    with pytest.raises(ValueError):
        build_alter_column_comment_sql(
            "workspace", "cbioportal", "patient", "col`name", "x",
        )


def test_build_alter_column_comment_sql_rejects_empty_identifier() -> None:
    with pytest.raises(ValueError):
        build_alter_column_comment_sql(
            "workspace", "cbioportal_x", "patient", "", "x",
        )


def test_build_alter_table_comment_sql_quotes_identifiers() -> None:
    sql = build_alter_table_comment_sql(
        "workspace", "cbioportal_x", "patient",
        "cBioPortal clinical patient from data_clinical_patient.txt",
    )
    assert sql == (
        "COMMENT ON TABLE `workspace`.`cbioportal_x`.`patient` "
        "IS 'cBioPortal clinical patient from data_clinical_patient.txt'"
    )


def test_build_alter_table_comment_sql_escapes_single_quotes() -> None:
    sql = build_alter_table_comment_sql(
        "workspace", "cbioportal_x", "patient", "Patient's table",
    )
    assert "'Patient''s table'" in sql


def test_build_alter_table_comment_sql_empty_comment_clears() -> None:
    sql = build_alter_table_comment_sql(
        "workspace", "cbioportal_x", "patient", "",
    )
    assert sql.endswith("IS ''")


def test_build_alter_table_comment_sql_rejects_semicolon_in_identifier() -> None:
    with pytest.raises(ValueError):
        build_alter_table_comment_sql(
            "workspace", "cbioportal_x", "pa;tient", "x",
        )
