from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sema.ingest.duckdb_staging import Staging
from sema.ingest.omop import (
    ingest_cdm_schema,
    load_field_level_comments,
    parse_postgres_ddl,
    postgres_to_duckdb_type,
)


@pytest.mark.unit
class TestPostgresTypeTranslation:
    @pytest.mark.parametrize(
        "pg_type,duckdb_type",
        [
            ("TIMESTAMP WITH TIME ZONE", "TIMESTAMPTZ"),
            ("timestamp with time zone", "TIMESTAMPTZ"),
            ("SERIAL", "INTEGER"),
            ("BIGSERIAL", "BIGINT"),
            ("TEXT", "VARCHAR"),
            ("VARCHAR(50)", "VARCHAR"),
            ("varchar(255)", "VARCHAR"),
            ("INTEGER", "INTEGER"),
            ("BIGINT", "BIGINT"),
            ("NUMERIC", "DOUBLE"),
            ("NUMERIC(10,2)", "DOUBLE"),
            ("DATE", "DATE"),
            ("TIMESTAMP", "TIMESTAMP"),
            ("BOOLEAN", "BOOLEAN"),
        ],
    )
    def test_translates_common_pg_types(self, pg_type: str, duckdb_type: str) -> None:
        assert postgres_to_duckdb_type(pg_type) == duckdb_type


@pytest.mark.unit
class TestPostgresDDLParser:
    def test_parses_create_table_statement(self) -> None:
        ddl = """
        CREATE TABLE person (
            person_id integer NOT NULL,
            gender_concept_id integer NOT NULL,
            year_of_birth integer NULL,
            birth_datetime TIMESTAMP,
            location_id bigint
        );
        """
        tables = parse_postgres_ddl(ddl)
        assert "person" in tables
        cols = {c.name: c for c in tables["person"]}
        assert cols["person_id"].postgres_type.upper() == "INTEGER"
        assert cols["person_id"].nullable is False
        assert cols["year_of_birth"].nullable is True
        assert cols["birth_datetime"].postgres_type.upper() == "TIMESTAMP"

    def test_parses_multiple_tables(self) -> None:
        ddl = """
        CREATE TABLE person (person_id integer NOT NULL);
        CREATE TABLE observation (
            observation_id integer NOT NULL,
            observation_date date NOT NULL
        );
        """
        tables = parse_postgres_ddl(ddl)
        assert set(tables.keys()) == {"person", "observation"}

    def test_handles_schema_qualified_names(self) -> None:
        ddl = """
        CREATE TABLE @cdmDatabaseSchema.person (
            person_id integer NOT NULL
        );
        """
        tables = parse_postgres_ddl(ddl)
        assert "person" in tables

    def test_ignores_constraints_and_alter_statements(self) -> None:
        ddl = """
        CREATE TABLE person (person_id integer NOT NULL);
        ALTER TABLE person ADD CONSTRAINT xpk_person PRIMARY KEY (person_id);
        """
        tables = parse_postgres_ddl(ddl)
        assert "person" in tables
        assert [c.name for c in tables["person"]] == ["person_id"]


@pytest.mark.unit
class TestFieldLevelCommentsLoader:
    def test_builds_table_column_comment_map(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "fields.csv"
        csv_path.write_text(
            "cdmTableName,cdmFieldName,userGuidance\n"
            "PERSON,person_id,A unique identifier for each person.\n"
            "PERSON,year_of_birth,Year of birth.\n"
            "OBSERVATION,observation_id,Obs PK\n",
            encoding="utf-8",
        )
        comments = load_field_level_comments(csv_path)
        assert comments[("person", "person_id")] == "A unique identifier for each person."
        assert comments[("person", "year_of_birth")] == "Year of birth."
        assert comments[("observation", "observation_id")] == "Obs PK"


@pytest.mark.unit
class TestIngestCdmSchema:
    def test_creates_empty_tables_with_types_and_comments(self, tmp_path: Path) -> None:
        staging = Staging(str(tmp_path / "staging.duckdb"))
        ddl = """
        CREATE TABLE person (
            person_id integer NOT NULL,
            birth_datetime TIMESTAMP WITH TIME ZONE
        );
        """
        fields_csv = tmp_path / "fields.csv"
        fields_csv.write_text(
            "cdmTableName,cdmFieldName,userGuidance\n"
            "PERSON,person_id,Primary key for the person table\n",
            encoding="utf-8",
        )
        with patch("sema.ingest.omop.fetch_cdm_artifacts") as mock_fetch:
            mock_fetch.return_value = (ddl, fields_csv)
            ingest_cdm_schema(version="5.4", staging=staging)

        info = staging.describe("ontology_omop", "person")
        assert info.columns["person_id"].type.upper().startswith("INT")
        assert info.columns["birth_datetime"].type.upper().startswith("TIMESTAMP")
        assert info.columns["person_id"].comment == "Primary key for the person table"

    def test_alternate_cdm_version_passes_tag_to_fetcher(self, tmp_path: Path) -> None:
        staging = Staging(str(tmp_path / "staging.duckdb"))
        ddl = "CREATE TABLE person (person_id integer NOT NULL);"
        fields_csv = tmp_path / "fields.csv"
        fields_csv.write_text("cdmTableName,cdmFieldName,userGuidance\n", encoding="utf-8")

        mock_fetch = MagicMock(return_value=(ddl, fields_csv))
        with patch("sema.ingest.omop.fetch_cdm_artifacts", mock_fetch):
            ingest_cdm_schema(version="5.3", staging=staging)

        mock_fetch.assert_called_once()
        called_version = mock_fetch.call_args.kwargs.get("version") or mock_fetch.call_args.args[0]
        assert called_version == "5.3"
