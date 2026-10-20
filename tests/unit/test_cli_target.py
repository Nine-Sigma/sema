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


_STAGING_SPEC = """\
concept_source:
  spec_version: 2
  id: land-codes
  identity_system: ex
  release: "1"
  default_provider: staging
  scheme: {qualified_name: ex.codes, description: Codes}
  concepts:
    relation: land.codes
    group_by: code
    roles: {concept_id: code, name: label}
"""


def _staging_file(tmp_path: Path) -> Path:
    import pyarrow as pa

    from sema.ingest.duckdb_staging import Staging

    path = tmp_path / "staging.duckdb"
    with Staging(str(path), schemas=("land",)) as staging:
        staging.write_table(
            "land",
            "codes",
            pa.table({"code": ["A", "B"], "label": ["Alpha", "Beta"]}),
            {"code": "VARCHAR", "label": "VARCHAR"},
            {},
            None,
        )
    return path


def _no_connector() -> None:
    raise AssertionError("a staging-only spec must not construct a connector")


def _import_args(tmp_path: Path, *extra: str) -> list[str]:
    spec = tmp_path / "spec.yaml"
    spec.write_text(_STAGING_SPEC)
    return [
        "import",
        "--spec",
        str(spec),
        "--out",
        str(tmp_path / "manifest.yaml"),
        "--evidence-out",
        str(tmp_path / "evidence.json"),
        *extra,
    ]


def test_target_import_staging_spec_never_constructs_a_connector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    monkeypatch.setattr("sema.cli_target._import_connector", _no_connector)
    staging = _staging_file(tmp_path)

    result = CliRunner().invoke(
        target_group, _import_args(tmp_path, "--staging", str(staging))
    )

    assert result.exit_code == 0, result.output
    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    names = {p["name"] for e in manifest["entities"] for p in e["properties"]}
    assert names == {"Alpha", "Beta"}


def test_target_import_staging_spec_without_staging_names_provider_and_relation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sema.cli_target._import_connector", _no_connector)

    result = CliRunner().invoke(target_group, _import_args(tmp_path))

    assert result.exit_code != 0
    assert "staging" in result.output
    assert "land.codes" in result.output
    assert not (tmp_path / "manifest.yaml").exists()
