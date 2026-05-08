"""CLI tests for `sema target load --manifest <path>`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from sema.cli_target import target_group
from sema.targets.registry import _clear_for_tests

pytestmark = pytest.mark.unit

_GOLDEN = (
    Path(__file__).resolve().parents[1]
    / "unit"
    / "targets"
    / "fixtures"
    / "golden_manifest.yaml"
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    _clear_for_tests()
    yield
    _clear_for_tests()


def test_target_load_in_memory_writer_prints_summary() -> None:
    runner = CliRunner()
    result = runner.invoke(
        target_group, ["load", "--manifest", str(_GOLDEN), "--writer", "in-memory"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["target_model_id"] == "golden-target"
    assert payload["target_schema_snapshot_hash"]
    assert "entities" in payload and len(payload["entities"]) > 0
    assert "context_cards" in payload and len(payload["context_cards"]) > 0


def test_target_load_rejects_missing_manifest() -> None:
    runner = CliRunner()
    result = runner.invoke(
        target_group, ["load", "--manifest", "/no/such/file.yaml", "--writer", "in-memory"]
    )
    assert result.exit_code != 0
    assert "manifest" in result.output.lower() or "no such" in result.output.lower()


def test_target_load_default_writer_is_in_memory_when_no_neo4j_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(target_group, ["load", "--manifest", str(_GOLDEN)])
    assert result.exit_code == 0, result.output


def test_target_load_skip_facets_threaded() -> None:
    runner = CliRunner()
    result = runner.invoke(
        target_group,
        [
            "load",
            "--manifest",
            str(_GOLDEN),
            "--writer",
            "in-memory",
            "--skip-facet",
            "semantic_aliases",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    statuses = [
        d["decisions"]["semantic_aliases"]["status"]
        for d in payload["enrichment_decisions"]
    ]
    assert any(s == "required_skipped" for s in statuses)
