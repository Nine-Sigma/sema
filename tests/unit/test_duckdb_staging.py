from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from sema.ingest.duckdb_staging import Staging


@pytest.fixture
def staging(tmp_path: Path) -> Staging:
    db_path = tmp_path / "test.duckdb"
    return Staging(str(db_path))


@pytest.mark.unit
class TestStagingSchemaLifecycle:
    def test_creates_file_and_schemas_on_init(self, tmp_path: Path) -> None:
        db_path = tmp_path / "new.duckdb"
        assert not db_path.exists()

        staging = Staging(str(db_path))

        assert db_path.exists()
        schemas = staging.list_schemas()
        assert "cbioportal" in schemas
        assert "ontology_omop" in schemas
        assert "vocabulary_omop" in schemas

    def test_expands_home_in_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        staging = Staging("~/poc.duckdb")
        assert (tmp_path / "poc.duckdb").exists()
        staging.close()

    def test_reopen_is_idempotent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "reopen.duckdb"
        Staging(str(db_path)).close()
        reopened = Staging(str(db_path))
        assert "cbioportal" in reopened.list_schemas()


@pytest.mark.unit
class TestStagingWriteTable:
    def test_writes_pyarrow_table_with_types(self, staging: Staging) -> None:
        rows = pa.table({"id": [1, 2], "name": ["alice", "bob"]})
        staging.write_table(
            schema="cbioportal",
            table="patient",
            rows=rows,
            column_types={"id": "INTEGER", "name": "VARCHAR"},
            column_comments={"id": "patient pk", "name": "display name"},
            table_comment="cbioportal patients",
        )

        info = staging.describe("cbioportal", "patient")
        assert info.columns["id"].type.upper().startswith("INT")
        assert info.columns["id"].comment == "patient pk"
        assert info.columns["name"].comment == "display name"
        assert info.table_comment == "cbioportal patients"

    def test_drop_and_recreate_is_idempotent(self, staging: Staging) -> None:
        rows = pa.table({"id": [1]})
        for _ in range(2):
            staging.write_table(
                schema="cbioportal",
                table="sample",
                rows=rows,
                column_types={"id": "INTEGER"},
                column_comments={},
                table_comment=None,
            )
        info = staging.describe("cbioportal", "sample")
        assert list(info.columns.keys()) == ["id"]

    def test_drop_table_removes_table(self, staging: Staging) -> None:
        rows = pa.table({"id": [1]})
        staging.write_table(
            schema="cbioportal",
            table="doomed",
            rows=rows,
            column_types={"id": "INTEGER"},
            column_comments={},
            table_comment=None,
        )
        staging.drop_table("cbioportal", "doomed")
        with pytest.raises(Exception):
            staging.describe("cbioportal", "doomed")

    def test_comment_with_single_quote_is_escaped(self, staging: Staging) -> None:
        rows = pa.table({"x": [1]})
        staging.write_table(
            schema="cbioportal",
            table="quoted",
            rows=rows,
            column_types={"x": "INTEGER"},
            column_comments={"x": "O'Brien's column"},
            table_comment="study with 'quotes' inside",
        )
        info = staging.describe("cbioportal", "quoted")
        assert info.columns["x"].comment == "O'Brien's column"
        assert info.table_comment == "study with 'quotes' inside"


@pytest.mark.unit
class TestStagingDescribe:
    def test_describe_uses_duckdb_metadata_functions(self, staging: Staging) -> None:
        rows = pa.table({"c": [1]})
        staging.write_table(
            schema="ontology_omop",
            table="person",
            rows=rows,
            column_types={"c": "BIGINT"},
            column_comments={"c": "person_id column"},
            table_comment="OMOP person table",
        )

        info = staging.describe("ontology_omop", "person")
        assert info.columns["c"].type.upper().startswith("BIG")
        assert info.columns["c"].comment == "person_id column"
        assert info.table_comment == "OMOP person table"
