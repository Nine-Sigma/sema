"""--no-quality-budget CLI flag (Section 6)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.unit


@pytest.fixture
def captured_config_holder():
    return {}


def _patch_run_build_capture(captured: dict, side_effect=None):
    def fake(cfg):
        captured["config"] = cfg
        if side_effect is not None:
            raise side_effect
        return {"tables_processed": 0, "failed_tables": []}
    return fake


class TestNoQualityBudgetFlag:
    def test_flag_sets_both_ceilings_to_one(
        self, captured_config_holder,
    ):
        from sema.cli import cli

        runner = CliRunner()
        with patch(
            "sema.cli.run_build",
            side_effect=_patch_run_build_capture(captured_config_holder),
        ):
            result = runner.invoke(cli, [
                "build", "--no-quality-budget", "--skip-embeddings",
            ])

        assert result.exit_code == 0, result.output
        cfg = captured_config_holder["config"]
        assert cfg.quality_budget_max_failure_rate == 1.0
        assert cfg.quality_budget_max_non_contributing_rate == 1.0

    def test_default_keeps_strict_thresholds(
        self, captured_config_holder,
    ):
        from sema.cli import cli

        runner = CliRunner()
        with patch(
            "sema.cli.run_build",
            side_effect=_patch_run_build_capture(captured_config_holder),
        ):
            result = runner.invoke(cli, [
                "build", "--skip-embeddings",
            ])

        assert result.exit_code == 0, result.output
        cfg = captured_config_holder["config"]
        assert cfg.quality_budget_max_failure_rate == 0.30
        assert cfg.quality_budget_max_non_contributing_rate == 0.40


class TestExitCodeOnBudgetExceeded:
    def test_exit_code_seven_on_quality_budget_exception(
        self, captured_config_holder,
    ):
        from sema.cli import cli
        from sema.pipeline.quality_budget import (
            Breakdown, QualityBudgetExceeded,
        )

        err = QualityBudgetExceeded(
            trigger="stage_b_failure_rate",
            failure_rate=0.40,
            threshold=0.30,
            denominator=10,
            breakdown=Breakdown(succeeded=6, stage_b_failed=4),
        )
        runner = CliRunner()
        with patch(
            "sema.cli.run_build",
            side_effect=_patch_run_build_capture(
                captured_config_holder, side_effect=err,
            ),
        ):
            result = runner.invoke(cli, [
                "build", "--skip-embeddings",
            ])

        assert result.exit_code == 7

    def test_exit_code_one_on_other_exception(
        self, captured_config_holder,
    ):
        from sema.cli import cli

        runner = CliRunner()
        with patch(
            "sema.cli.run_build",
            side_effect=_patch_run_build_capture(
                captured_config_holder,
                side_effect=RuntimeError("oops"),
            ),
        ):
            result = runner.invoke(cli, [
                "build", "--skip-embeddings",
            ])

        assert result.exit_code == 1
