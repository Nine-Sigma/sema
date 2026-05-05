from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from sema.ingest.databricks_push import Bridge, PushError, PushResult
from sema.ingest.duckdb_staging import Staging
from sema.ingest.study_registry import StudyRegistry
from sema.models.config import (
    DatabricksConfig,
    IngestConfig,
    IngestDatabricksTargetConfig,
)
from pydantic import SecretStr


def _config(cloud_uri: str | None = None) -> IngestConfig:
    creds = DatabricksConfig(
        host="https://test.databricks.com",
        token=SecretStr("token"),
        http_path="/sql/1.0/warehouses/abc",
    )
    target = IngestDatabricksTargetConfig(catalog="workspace")
    return IngestConfig(
        databricks=target,
        databricks_creds=creds,
        cloud_staging_uri=cloud_uri,
    )


def _mock_cursor() -> MagicMock:
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    return cursor


def _mock_connection(cursor: MagicMock) -> MagicMock:
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


@pytest.fixture
def staging(tmp_path: Path) -> Staging:
    s = Staging(str(tmp_path / "bridge.duckdb"))
    s.execute('CREATE SCHEMA IF NOT EXISTS "cbioportal"')
    rows = pa.table({"patient_id": ["P-1", "P-2"], "age": [40, 50]})
    s.write_table(
        schema="cbioportal",
        table="patient",
        rows=rows,
        column_types={"patient_id": "VARCHAR", "age": "INTEGER"},
        column_comments={"patient_id": "patient primary key"},
        table_comment="cBioPortal patients",
    )
    return s


@pytest.mark.unit
class TestBridgeProvisioning:
    def test_ensure_schemas_issues_create_for_registered_and_shared(
        self, staging: Staging
    ) -> None:
        registry = StudyRegistry(staging)
        registry.register("cbioportal_msk_chord_2024", "msk_chord_2024", "cbioportal")
        cursor = _mock_cursor()
        conn = _mock_connection(cursor)
        with patch("sema.ingest.databricks_push.sql_connect", return_value=conn):
            bridge = Bridge(_config(), staging=staging)
            bridge.ensure_schemas()

        executed = [call.args[0] for call in cursor.execute.call_args_list]
        assert any(
            "CREATE SCHEMA IF NOT EXISTS `workspace`.`cbioportal_msk_chord_2024`" in sql
            for sql in executed
        )
        assert any("CREATE SCHEMA IF NOT EXISTS `workspace`.`ontology_omop`" in sql for sql in executed)
        assert any("CREATE SCHEMA IF NOT EXISTS `workspace`.`vocabulary_omop`" in sql for sql in executed)


@pytest.mark.unit
class TestDdlFromDuckDB:
    def test_generates_create_table_with_types_and_comments(self, staging: Staging) -> None:
        cursor = _mock_cursor()
        conn = _mock_connection(cursor)
        with patch("sema.ingest.databricks_push.sql_connect", return_value=conn):
            bridge = Bridge(_config(), staging=staging)
            ddl = bridge._ddl_from_duckdb("cbioportal", "patient")

        assert "CREATE TABLE `workspace`.`cbioportal`.`patient`" in ddl
        assert "`patient_id` STRING" in ddl
        assert "`age` INT" in ddl
        assert "COMMENT 'patient primary key'" in ddl
        assert "COMMENT 'cBioPortal patients'" in ddl


@pytest.mark.unit
class TestPushViaInsert:
    def test_inserts_rows_in_batches(self, staging: Staging) -> None:
        cursor = _mock_cursor()
        cursor.fetchone.return_value = (2,)
        conn = _mock_connection(cursor)
        with patch("sema.ingest.databricks_push.sql_connect", return_value=conn):
            bridge = Bridge(_config(), staging=staging)
            result = bridge.push_table("cbioportal", "patient")

        insert_calls = [
            call.args[0]
            for call in cursor.execute.call_args_list
            if call.args[0].strip().startswith("INSERT INTO")
        ]
        assert len(insert_calls) >= 1
        assert "VALUES" in insert_calls[0]
        assert result.rows_pushed == 2
        assert result.mechanism == "insert"


@pytest.mark.unit
class TestCopyIntoRouting:
    def test_routes_large_vocab_tables_via_copy_into(self, tmp_path: Path) -> None:
        staging = Staging(str(tmp_path / "route.duckdb"))
        rows = pa.table({"concept_id": [1, 2]})
        staging.write_table(
            schema="vocabulary_omop",
            table="concept_ancestor",
            rows=rows,
            column_types={"concept_id": "BIGINT"},
            column_comments={},
            table_comment=None,
        )
        cursor = _mock_cursor()
        cursor.fetchone.return_value = (2,)
        conn = _mock_connection(cursor)

        cloud_uri = f"file://{tmp_path}/staging"
        with patch("sema.ingest.databricks_push.sql_connect", return_value=conn):
            bridge = Bridge(_config(cloud_uri=cloud_uri), staging=staging)
            result = bridge.push_table("vocabulary_omop", "concept_ancestor")

        copy_calls = [
            call.args[0]
            for call in cursor.execute.call_args_list
            if call.args[0].strip().startswith("COPY INTO")
        ]
        assert len(copy_calls) == 1
        assert "vocabulary_omop" in copy_calls[0]
        assert "concept_ancestor" in copy_calls[0]
        assert result.mechanism == "copy_into"

    def test_falls_back_to_insert_when_no_staging_uri(self, tmp_path: Path) -> None:
        staging = Staging(str(tmp_path / "fallback.duckdb"))
        rows = pa.table({"concept_id": [1]})
        staging.write_table(
            schema="vocabulary_omop",
            table="concept_ancestor",
            rows=rows,
            column_types={"concept_id": "BIGINT"},
            column_comments={},
            table_comment=None,
        )
        cursor = _mock_cursor()
        cursor.fetchone.return_value = (1,)
        conn = _mock_connection(cursor)
        with patch("sema.ingest.databricks_push.sql_connect", return_value=conn):
            bridge = Bridge(_config(cloud_uri=None), staging=staging)
            result = bridge.push_table("vocabulary_omop", "concept_ancestor")

        assert result.mechanism == "insert"


@pytest.mark.unit
class TestPushSchemasErrorHandling:
    def test_one_table_failure_continues_with_others(self, tmp_path: Path) -> None:
        staging = Staging(str(tmp_path / "errors.duckdb"))
        staging.execute('CREATE SCHEMA IF NOT EXISTS "cbioportal"')
        for name in ["patient", "sample"]:
            staging.write_table(
                schema="cbioportal",
                table=name,
                rows=pa.table({"id": [1]}),
                column_types={"id": "INTEGER"},
                column_comments={},
                table_comment=None,
            )
        cursor = _mock_cursor()
        cursor.fetchone.return_value = (1,)
        call_count = {"n": 0}

        def execute_side_effect(sql: str, *_: object) -> None:
            call_count["n"] += 1
            if "INSERT INTO `workspace`.`cbioportal`.`patient`" in sql:
                raise RuntimeError("boom")

        cursor.execute.side_effect = execute_side_effect
        conn = _mock_connection(cursor)

        with patch("sema.ingest.databricks_push.sql_connect", return_value=conn):
            bridge = Bridge(_config(), staging=staging)
            with pytest.raises(PushError) as exc:
                bridge.push_schemas(["cbioportal"])

        assert "patient" in str(exc.value)
        assert "sample" not in str(exc.value)


@pytest.mark.unit
class TestRowCountVerification:
    def test_logs_warning_on_count_mismatch(self, staging: Staging, caplog: pytest.LogCaptureFixture) -> None:
        cursor = _mock_cursor()
        cursor.fetchone.return_value = (99,)
        conn = _mock_connection(cursor)
        with patch("sema.ingest.databricks_push.sql_connect", return_value=conn):
            bridge = Bridge(_config(), staging=staging)
            result: PushResult = bridge.push_table("cbioportal", "patient")

        assert result.rows_pushed == 2
        assert result.target_count == 99
        assert result.count_mismatch is True


@pytest.mark.unit
class TestUcVolumeDetection:
    def test_is_uc_volume_path_true_for_volumes_prefix(self) -> None:
        from sema.ingest.databricks_push_utils import is_uc_volume_path

        assert is_uc_volume_path("/Volumes/workspace/default/sema_staging") is True
        assert is_uc_volume_path("/Volumes/workspace/default/sema_staging/") is True

    def test_is_uc_volume_path_false_for_other_uris(self) -> None:
        from sema.ingest.databricks_push_utils import is_uc_volume_path

        assert is_uc_volume_path("file:///tmp/foo") is False
        assert is_uc_volume_path("s3://bucket/key") is False
        assert is_uc_volume_path("/tmp/foo") is False
        assert is_uc_volume_path("dbfs:/tmp/foo") is False


@pytest.mark.unit
class TestSizeBasedRouting:
    def test_route_via_copy_into_when_above_threshold(self) -> None:
        from sema.ingest.databricks_push_utils import (
            COPY_INTO_ROW_THRESHOLD,
            should_route_via_copy_into,
        )

        assert should_route_via_copy_into(
            "cbioportal_msk_chord_2024", "cna",
            row_count=COPY_INTO_ROW_THRESHOLD,
        ) is True

    def test_no_copy_into_when_below_threshold_and_not_in_allowlist(self) -> None:
        from sema.ingest.databricks_push_utils import should_route_via_copy_into

        assert should_route_via_copy_into(
            "cbioportal_msk_chord_2024", "patient", row_count=24_950,
        ) is False

    def test_allowlist_entries_route_regardless_of_row_count(self) -> None:
        from sema.ingest.databricks_push_utils import should_route_via_copy_into

        assert should_route_via_copy_into(
            "vocabulary_omop", "concept_ancestor", row_count=0,
        ) is True


@pytest.mark.unit
class TestUcVolumeUpload:
    def test_uc_volume_uri_uploads_via_workspace_client(self, tmp_path: Path) -> None:
        staging = Staging(str(tmp_path / "uc.duckdb"))
        staging.execute('CREATE SCHEMA IF NOT EXISTS "cbioportal_msk_chord_2024"')
        rows = pa.table({"id": list(range(150_000))})
        staging.write_table(
            schema="cbioportal_msk_chord_2024",
            table="cna",
            rows=rows,
            column_types={"id": "INTEGER"},
            column_comments={},
            table_comment=None,
        )
        cursor = _mock_cursor()
        cursor.fetchone.return_value = (150_000,)
        conn = _mock_connection(cursor)

        ws_client = MagicMock()
        with patch("sema.ingest.databricks_push.sql_connect", return_value=conn), \
             patch("sema.ingest.databricks_push.WorkspaceClient", return_value=ws_client):
            cfg = _config(cloud_uri="/Volumes/workspace/default/sema_staging")
            bridge = Bridge(cfg, staging=staging)
            result = bridge.push_table("cbioportal_msk_chord_2024", "cna")

        assert result.mechanism == "copy_into"
        assert ws_client.files.upload.call_count == 1
        upload_kwargs = ws_client.files.upload.call_args.kwargs
        assert upload_kwargs["file_path"].startswith(
            "/Volumes/workspace/default/sema_staging/cbioportal_msk_chord_2024/cna/"
        )
        assert upload_kwargs["overwrite"] is True
