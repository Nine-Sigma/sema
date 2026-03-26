import json

import pytest
from click.testing import CliRunner
from unittest.mock import MagicMock, patch, AsyncMock

pytestmark = pytest.mark.unit

from sema.cli import cli
from sema.pipeline.orchestrate import run_build
from sema.models.config import BuildConfig


@pytest.fixture
def runner():
    return CliRunner()


class TestBuildCommand:
    def test_build_calls_pipeline_in_order(self, runner):
        with patch("sema.cli.run_build") as mock_build:
            mock_build.return_value = {
                "tables_processed": 5,
                "entities_created": 5,
                "properties_created": 20,
                "value_sets_created": 8,
                "terms_created": 45,
                "joins_inferred": 4,
                "confidence_distribution": {"high": 15, "medium": 10, "low": 5},
            }
            result = runner.invoke(cli, [
                "build",
                "--source", "databricks",
                "--catalog", "cdm",
                "--schemas", "clinical,staging",
            ])
            assert result.exit_code == 0
            mock_build.assert_called_once()
            call_config = mock_build.call_args[0][0]
            assert call_config.source == "databricks"
            assert call_config.catalog == "cdm"
            assert call_config.schemas == ["clinical", "staging"]

    def test_build_prints_report(self, runner):
        with patch("sema.cli.run_build") as mock_build:
            mock_build.return_value = {
                "tables_processed": 3,
                "entities_created": 3,
                "properties_created": 12,
                "value_sets_created": 4,
                "terms_created": 20,
                "joins_inferred": 2,
                "confidence_distribution": {"high": 10, "medium": 5, "low": 2},
            }
            result = runner.invoke(cli, ["build", "--source", "databricks", "--catalog", "test"])
            assert result.exit_code == 0
            assert "Tables Processed" in result.output

    def test_build_failure_prints_error(self, runner):
        with patch("sema.cli.run_build") as mock_build:
            mock_build.side_effect = RuntimeError("Connection refused to Databricks")
            result = runner.invoke(cli, ["build", "--source", "databricks", "--catalog", "test"])
            assert result.exit_code != 0
            assert "Connection refused" in result.output

    def test_build_verbose(self, runner):
        with patch("sema.cli.run_build") as mock_build:
            mock_build.return_value = {
                "tables_processed": 1,
                "entities_created": 1,
                "properties_created": 3,
                "value_sets_created": 1,
                "terms_created": 5,
                "joins_inferred": 0,
                "confidence_distribution": {"high": 3, "medium": 1, "low": 1},
            }
            result = runner.invoke(cli, ["build", "--source", "databricks", "--catalog", "test", "--verbose"])
            assert result.exit_code == 0
            call_config = mock_build.call_args[0][0]
            assert call_config.verbose is True

    def test_build_with_config_file(self, runner, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("source: databricks\ncatalog: from_file\n")
        with patch("sema.cli.run_build") as mock_build:
            mock_build.return_value = {"tables_processed": 0, "entities_created": 0, "properties_created": 0, "value_sets_created": 0, "terms_created": 0, "joins_inferred": 0, "confidence_distribution": {}}
            result = runner.invoke(cli, ["build", "--config", str(config_file)])
            assert result.exit_code == 0
            call_config = mock_build.call_args[0][0]
            assert call_config.catalog == "from_file"

    def test_build_flags_override_config_file(self, runner, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("source: databricks\ncatalog: from_file\n")
        with patch("sema.cli.run_build") as mock_build:
            mock_build.return_value = {"tables_processed": 0, "entities_created": 0, "properties_created": 0, "value_sets_created": 0, "terms_created": 0, "joins_inferred": 0, "confidence_distribution": {}}
            result = runner.invoke(cli, ["build", "--config", str(config_file), "--catalog", "override"])
            assert result.exit_code == 0
            call_config = mock_build.call_args[0][0]
            assert call_config.catalog == "override"


class TestContextCommand:
    def test_context_produces_sco_json(self, runner):
        with patch("sema.cli.run_context") as mock_context:
            mock_context.return_value = {
                "entities": [{"name": "Cancer Diagnosis", "confidence": 0.85}],
                "properties": [{"name": "Stage", "semantic_type": "categorical"}],
                "physical_assets": {"tables": ["cancer_diagnosis"]},
                "join_paths": [],
                "governed_values": {},
            }
            result = runner.invoke(cli, ["context", "--question", "stage 3 colorectal patients"])
            assert result.exit_code == 0
            output = json.loads(result.output)
            assert "entities" in output

    def test_context_with_neo4j_uri(self, runner):
        with patch("sema.cli.run_context") as mock_context:
            mock_context.return_value = {"entities": [], "properties": [], "physical_assets": {}, "join_paths": [], "governed_values": {}}
            result = runner.invoke(cli, [
                "context",
                "--question", "test",
                "--neo4j-uri", "bolt://custom:7687",
            ])
            assert result.exit_code == 0
            call_config = mock_context.call_args[0][0]
            assert call_config.neo4j.uri == "bolt://custom:7687"


class TestQueryCommand:
    def test_query_default_plan_mode(self, runner):
        with patch("sema.cli.run_query") as mock_query:
            mock_query.return_value = {
                "mode": "plan",
                "sql": "SELECT * FROM cancer_diagnosis WHERE stage = 'Stage III'",
                "validation": {"valid": True, "errors": []},
            }
            result = runner.invoke(cli, ["query", "--question", "stage 3 patients"])
            assert result.exit_code == 0
            call_config = mock_query.call_args[0][0]
            assert call_config.mode.value == "plan"
            assert "SELECT" in result.output

    def test_query_execute_mode(self, runner):
        with patch("sema.cli.run_query") as mock_query:
            mock_query.return_value = {
                "mode": "execute",
                "sql": "SELECT * FROM cancer_diagnosis",
                "validation": {"valid": True, "errors": []},
                "results": [{"patient_id": "P1", "stage": "Stage III"}],
                "row_count": 1,
            }
            result = runner.invoke(cli, ["query", "--question", "test", "--mode", "execute"])
            assert result.exit_code == 0
            call_config = mock_query.call_args[0][0]
            assert call_config.mode.value == "execute"

    def test_query_explain_mode(self, runner):
        with patch("sema.cli.run_query") as mock_query:
            mock_query.return_value = {
                "mode": "explain",
                "sql": "SELECT * FROM cancer_diagnosis",
                "validation": {"valid": True, "errors": []},
                "explain": "== Physical Plan ==\nScan parquet",
            }
            result = runner.invoke(cli, ["query", "--question", "test", "--mode", "explain"])
            assert result.exit_code == 0

    def test_query_verbose(self, runner):
        with patch("sema.cli.run_query") as mock_query:
            mock_query.return_value = {
                "mode": "plan",
                "sql": "SELECT 1",
                "validation": {"valid": True, "errors": []},
            }
            result = runner.invoke(cli, ["query", "--question", "test", "--verbose"])
            assert result.exit_code == 0
            call_config = mock_query.call_args[0][0]
            assert call_config.verbose is True

    def test_query_invalid_mode(self, runner):
        result = runner.invoke(cli, ["query", "--question", "test", "--mode", "invalid"])
        assert result.exit_code != 0


class TestConfigLoading:
    def test_env_vars_used_when_no_flags(self, runner, monkeypatch):
        monkeypatch.setenv("NEO4J_URI", "bolt://from-env:7687")
        with patch("sema.cli.run_context") as mock_context:
            mock_context.return_value = {"entities": [], "properties": [], "physical_assets": {}, "join_paths": [], "governed_values": {}}
            result = runner.invoke(cli, ["context", "--question", "test"], env={"NEO4J_URI": "bolt://from-env:7687"})
            assert result.exit_code == 0
            call_config = mock_context.call_args[0][0]
            assert call_config.neo4j.uri == "bolt://from-env:7687"

    def test_flags_override_env_vars(self, runner, monkeypatch):
        monkeypatch.setenv("NEO4J_URI", "bolt://from-env:7687")
        with patch("sema.cli.run_context") as mock_context:
            mock_context.return_value = {"entities": [], "properties": [], "physical_assets": {}, "join_paths": [], "governed_values": {}}
            result = runner.invoke(cli, ["context", "--question", "test", "--neo4j-uri", "bolt://from-flag:7687"], env={"NEO4J_URI": "bolt://from-env:7687"})
            assert result.exit_code == 0
            call_config = mock_context.call_args[0][0]
            assert call_config.neo4j.uri == "bolt://from-flag:7687"

    def test_llm_config_flags(self, runner):
        with patch("sema.cli.run_build") as mock_build:
            mock_build.return_value = {"tables_processed": 0, "entities_created": 0, "properties_created": 0, "value_sets_created": 0, "terms_created": 0, "joins_inferred": 0, "confidence_distribution": {}}
            result = runner.invoke(cli, [
                "build",
                "--source", "databricks",
                "--catalog", "test",
                "--llm-provider", "openai",
                "--llm-model", "gpt-4o",
            ])
            assert result.exit_code == 0
            call_config = mock_build.call_args[0][0]
            assert call_config.llm.provider == "openai"
            assert call_config.llm.model == "gpt-4o"


class TestRunBuildCharacterization:
    """Characterization tests capturing current behavior of run_build."""

    @patch("sema.pipeline.orchestrate._get_embedder")
    @patch("sema.pipeline.orchestrate._get_llm")
    @patch("sema.pipeline.orchestrate._get_neo4j_driver")
    @patch("sema.connectors.databricks.sql_connect")
    def test_run_build_processes_tables_and_returns_report(
        self,
        mock_sql_connect,
        mock_get_neo4j,
        mock_get_llm,
        mock_get_embedder,
    ):
        from sema.connectors.databricks import TableWorkItem
        from sema.pipeline.build import TableResult

        # Prevent real DB connections
        mock_sql_connect.return_value = MagicMock()

        # Mock driver
        mock_driver = MagicMock()
        mock_get_neo4j.return_value = mock_driver

        # Mock LLM
        mock_get_llm.return_value = MagicMock()

        # Mock embedder (raise so it skips embedding step)
        mock_get_embedder.side_effect = Exception("no embedder configured")

        work_items = [
            TableWorkItem(catalog="cdm", schema="clinical", table_name="patients", fqn="unity://cdm.clinical.patients"),
            TableWorkItem(catalog="cdm", schema="clinical", table_name="visits", fqn="unity://cdm.clinical.visits"),
        ]

        with patch(
            "sema.connectors.databricks.DatabricksConnector.discover_tables",
            return_value=work_items,
        ), patch(
            "sema.pipeline.build.process_table",
        ) as mock_process, patch(
            "sema.graph.loader.GraphLoader",
        ):
            mock_process.side_effect = [
                TableResult.success("unity://cdm.clinical.patients", entities=1, properties=5, terms=3),
                TableResult.success("unity://cdm.clinical.visits", entities=1, properties=4, terms=2),
            ]

            config = BuildConfig(
                source="databricks",
                catalog="cdm",
                schemas=["clinical"],
            )
            result = run_build(config)

        # Returns a dict with tables_processed key
        assert isinstance(result, dict)
        assert "tables_processed" in result
        assert result["tables_processed"] == 2
        assert result["entities_created"] == 2
        assert result["properties_created"] == 9
        assert result["terms_created"] == 5
        assert result["tables_failed"] == 0

        # Driver was closed
        mock_driver.close.assert_called_once()
