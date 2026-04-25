from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest
from pydantic import SecretStr

from sema.ingest.databricks_push import Bridge
from sema.ingest.duckdb_staging import Staging
from sema.ingest.study_registry import StudyRegistry
from sema.models.config import (
    DatabricksConfig,
    IngestConfig,
    IngestDatabricksTargetConfig,
)


def _config(target_schemas: list[str] | None = None) -> IngestConfig:
    creds = DatabricksConfig(
        host="https://test.databricks.com",
        token=SecretStr("token"),
        http_path="/sql/1.0/warehouses/abc",
    )
    target = IngestDatabricksTargetConfig(catalog="workspace")
    if target_schemas is not None:
        target = IngestDatabricksTargetConfig(catalog="workspace", schemas=target_schemas)
    return IngestConfig(databricks=target, databricks_creds=creds)


def _mock_cursor() -> MagicMock:
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.fetchone.return_value = (1,)
    return cursor


def _mock_connection(cursor: MagicMock) -> MagicMock:
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


def _seed_table(staging: Staging, schema: str, table: str = "patient") -> None:
    staging.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    staging.write_table(
        schema=schema,
        table=table,
        rows=pa.table({"id": [1]}),
        column_types={"id": "INTEGER"},
        column_comments={},
        table_comment=None,
    )


@pytest.mark.unit
class TestStagingListRegisteredSchemas:
    def test_returns_shared_only_when_registry_absent(self, tmp_path: Path) -> None:
        staging = Staging(str(tmp_path / "no_reg.duckdb"))
        result = staging.list_registered_schemas()
        assert sorted(result) == ["ontology_omop", "vocabulary_omop"]

    def test_unions_registry_with_shared(self, tmp_path: Path) -> None:
        staging = Staging(str(tmp_path / "reg.duckdb"))
        registry = StudyRegistry(staging)
        registry.register("cbioportal_msk_chord_2024", "msk_chord_2024", "cbioportal")
        result = staging.list_registered_schemas()
        assert sorted(result) == [
            "cbioportal_msk_chord_2024",
            "ontology_omop",
            "vocabulary_omop",
        ]

    def test_dedups_when_registry_lists_shared_schema(self, tmp_path: Path) -> None:
        staging = Staging(str(tmp_path / "dedup.duckdb"))
        registry = StudyRegistry(staging)
        registry.register("ontology_omop", "shared_ontology", "omop")
        result = staging.list_registered_schemas()
        assert result.count("ontology_omop") == 1


@pytest.mark.unit
class TestStagingListAllSchemas:
    def test_excludes_system_schemas(self, tmp_path: Path) -> None:
        staging = Staging(str(tmp_path / "sys.duckdb"))
        result = staging.list_all_schemas()
        assert "main" not in result
        assert "information_schema" not in result
        assert "pg_catalog" not in result
        assert "_sema" not in result
        assert "ontology_omop" in result
        assert "vocabulary_omop" in result

    def test_includes_user_created_schemas(self, tmp_path: Path) -> None:
        staging = Staging(str(tmp_path / "user.duckdb"))
        staging.execute('CREATE SCHEMA "scratch_experiment"')
        result = staging.list_all_schemas()
        assert "scratch_experiment" in result


@pytest.mark.unit
class TestPushSchemasDefaults:
    def test_default_uses_registry_and_shared_only(self, tmp_path: Path) -> None:
        staging = Staging(str(tmp_path / "default.duckdb"))
        registry = StudyRegistry(staging)
        registry.register("cbioportal_msk_chord_2024", "msk_chord_2024", "cbioportal")
        _seed_table(staging, "cbioportal_msk_chord_2024")
        _seed_table(staging, "ontology_omop")
        # scratch schema NOT registered; must NOT be pushed by default
        _seed_table(staging, "my_experiments")

        cursor = _mock_cursor()
        conn = _mock_connection(cursor)
        with patch("sema.ingest.databricks_push.sql_connect", return_value=conn):
            bridge = Bridge(_config(target_schemas=[]), staging=staging)
            results = bridge.push_schemas()

        pushed_schemas = {r.schema for r in results}
        assert "cbioportal_msk_chord_2024" in pushed_schemas
        assert "ontology_omop" in pushed_schemas
        assert "my_experiments" not in pushed_schemas

    def test_explicit_schemas_overrides_discovery(self, tmp_path: Path) -> None:
        staging = Staging(str(tmp_path / "explicit.duckdb"))
        registry = StudyRegistry(staging)
        registry.register("cbioportal_a", "A", "cbioportal")
        registry.register("cbioportal_b", "B", "cbioportal")
        _seed_table(staging, "cbioportal_a")
        _seed_table(staging, "cbioportal_b")

        cursor = _mock_cursor()
        conn = _mock_connection(cursor)
        with patch("sema.ingest.databricks_push.sql_connect", return_value=conn):
            bridge = Bridge(_config(target_schemas=[]), staging=staging)
            results = bridge.push_schemas(schemas=["cbioportal_a"])

        pushed_schemas = {r.schema for r in results}
        assert pushed_schemas == {"cbioportal_a"}

    def test_config_target_schemas_acts_as_filter(self, tmp_path: Path) -> None:
        staging = Staging(str(tmp_path / "filter.duckdb"))
        registry = StudyRegistry(staging)
        registry.register("cbioportal_a", "A", "cbioportal")
        registry.register("cbioportal_b", "B", "cbioportal")
        _seed_table(staging, "cbioportal_a")
        _seed_table(staging, "cbioportal_b")
        _seed_table(staging, "ontology_omop")

        cursor = _mock_cursor()
        conn = _mock_connection(cursor)
        with patch("sema.ingest.databricks_push.sql_connect", return_value=conn):
            bridge = Bridge(_config(target_schemas=["cbioportal_a"]), staging=staging)
            results = bridge.push_schemas()

        pushed_schemas = {r.schema for r in results}
        assert pushed_schemas == {"cbioportal_a"}

    def test_discover_all_schemas_includes_unregistered(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        staging = Staging(str(tmp_path / "all.duckdb"))
        registry = StudyRegistry(staging)
        registry.register("cbioportal_a", "A", "cbioportal")
        _seed_table(staging, "cbioportal_a")
        _seed_table(staging, "scratch_local")

        cursor = _mock_cursor()
        conn = _mock_connection(cursor)
        with patch("sema.ingest.databricks_push.sql_connect", return_value=conn):
            bridge = Bridge(_config(target_schemas=[]), staging=staging)
            results = bridge.push_schemas(discover_all=True)

        pushed_schemas = {r.schema for r in results}
        assert "cbioportal_a" in pushed_schemas
        assert "scratch_local" in pushed_schemas
