from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from sema.cli import cli


@pytest.mark.unit
class TestIngestCbioportalCommand:
    def test_calls_ingest_study_with_parsed_args(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with patch(
            "showcase.cbioportal_to_omop.parsers.ingest_study"
        ) as mock_ingest, patch("sema.cli_ingest.Staging") as mock_staging:
            mock_staging.return_value = MagicMock()
            result = runner.invoke(
                cli,
                [
                    "ingest", "cbioportal", "brca_tcga_pan_can_atlas_2018",
                    "--cache-dir", str(tmp_path / "cache"),
                    "--duckdb-path", str(tmp_path / "poc.duckdb"),
                ],
            )

        assert result.exit_code == 0, result.output
        mock_ingest.assert_called_once()
        kwargs = mock_ingest.call_args.kwargs
        assert kwargs["study_id"] == "brca_tcga_pan_can_atlas_2018"
        assert str(kwargs["cache_dir"]).endswith("cache")


@pytest.mark.unit
class TestIngestOmopCommand:
    def test_runs_cdm_only_when_vocab_path_not_provided(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with patch("sema.cli_ingest.ingest_cdm_schema") as mock_cdm, patch(
            "sema.cli_ingest.ingest_vocabulary"
        ) as mock_vocab, patch("sema.cli_ingest.Staging") as mock_staging:
            mock_staging.return_value = MagicMock()
            result = runner.invoke(
                cli,
                ["ingest", "omop", "--duckdb-path", str(tmp_path / "poc.duckdb")],
            )

        assert result.exit_code == 0, result.output
        mock_cdm.assert_called_once()
        mock_vocab.assert_called_once()
        assert mock_vocab.call_args.kwargs["vocab_path"] is None

    def test_runs_cdm_and_vocab_when_vocab_path_given(self, tmp_path: Path) -> None:
        runner = CliRunner()
        vocab_dir = tmp_path / "athena"
        vocab_dir.mkdir()
        with patch("sema.cli_ingest.ingest_cdm_schema") as mock_cdm, patch(
            "sema.cli_ingest.ingest_vocabulary"
        ) as mock_vocab, patch("sema.cli_ingest.Staging") as mock_staging:
            mock_staging.return_value = MagicMock()
            result = runner.invoke(
                cli,
                [
                    "ingest", "omop",
                    "--cdm-version", "5.3",
                    "--vocab-path", str(vocab_dir),
                    "--duckdb-path", str(tmp_path / "poc.duckdb"),
                ],
            )

        assert result.exit_code == 0, result.output
        assert mock_cdm.call_args.kwargs["version"] == "5.3"
        assert str(mock_vocab.call_args.kwargs["vocab_path"]) == str(vocab_dir)


@pytest.mark.unit
class TestPushCommand:
    def test_calls_push_schemas_with_all_by_default(self, tmp_path: Path) -> None:
        runner = CliRunner()
        bridge = MagicMock()
        bridge.push_schemas.return_value = []
        with patch("sema.cli_ingest.Bridge", return_value=bridge), patch(
            "sema.cli_ingest.Staging"
        ) as mock_staging:
            mock_staging.return_value = MagicMock()
            result = runner.invoke(
                cli, ["push", "--duckdb-path", str(tmp_path / "poc.duckdb")]
            )

        assert result.exit_code == 0, result.output
        bridge.push_schemas.assert_called_once_with(None)

    def test_scopes_to_requested_schemas(self, tmp_path: Path) -> None:
        runner = CliRunner()
        bridge = MagicMock()
        bridge.push_schemas.return_value = []
        with patch("sema.cli_ingest.Bridge", return_value=bridge), patch(
            "sema.cli_ingest.Staging"
        ) as mock_staging:
            mock_staging.return_value = MagicMock()
            result = runner.invoke(
                cli,
                [
                    "push", "--schemas", "cbioportal",
                    "--duckdb-path", str(tmp_path / "poc.duckdb"),
                ],
            )

        assert result.exit_code == 0, result.output
        bridge.push_schemas.assert_called_once_with(["cbioportal"])

    def test_unsupported_target_fails(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "push", "--target", "bigquery",
                "--duckdb-path", str(tmp_path / "poc.duckdb"),
            ],
        )
        assert result.exit_code != 0
        assert "bigquery" in result.output.lower() or "unsupported" in result.output.lower()
