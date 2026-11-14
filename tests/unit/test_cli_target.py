"""CLI tests for `sema target load`, `import`, and `project`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from sema.cli_target import target_group
from sema.graph.loader import GraphLoader
from sema.graph.physical_catalog import PhysicalColumn, StaticPhysicalCatalogSource
from sema.graph.target_join_paths import DECLARED_JOIN_CONFIDENCE
from sema.targets.registry import _clear_for_tests
from tests.unit.intake.test_acris_import import (
    ENTITIES,
    FOREIGN_KEYS,
    _Imported,
    imported,
)

pytestmark = pytest.mark.unit

__all__ = ["imported"]

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


# `sema target project`: the catalog and the loader come from two module
# factories, `_project_catalog_source` and `_project_loader(writer)`, which the
# tests replace the same way the import tests replace `_import_connector`.

_CATALOG = "cat"
_SCHEMA = "sch"


def _table_rows(imported: _Imported) -> list[dict[str, Any]]:
    return [e for e in imported.result.manifest["entities"] if e["kind"] == "TABLE_ROW"]


def _tables(imported: _Imported) -> dict[str, str]:
    names = sorted(e["qualified_name"] for e in _table_rows(imported))
    return {name: f"table_{i:02d}" for i, name in enumerate(names)}


def _static_catalog(imported: _Imported) -> StaticPhysicalCatalogSource:
    tables = _tables(imported)
    return StaticPhysicalCatalogSource(
        {
            (_CATALOG, _SCHEMA, tables[e["qualified_name"]]): [
                PhysicalColumn(name=p["name"], data_type="string", nullable=True)
                for p in e["properties"]
            ]
            for e in _table_rows(imported)
        }
    )


def _project_args(
    tmp_path: Path, imported: _Imported, entity_tables: dict[str, str] | None
) -> list[str]:
    import yaml

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(imported.result.manifest, sort_keys=False))
    args = ["project", "--manifest", str(manifest), "--writer", "in-memory"]
    if entity_tables is None:
        return args
    binding = tmp_path / "binding.yaml"
    binding.write_text(
        yaml.safe_dump(
            {"catalog": _CATALOG, "schema": _SCHEMA, "entity_tables": entity_tables},
            sort_keys=False,
        )
    )
    return [*args, "--binding", str(binding)]


def _fake_project(
    monkeypatch: pytest.MonkeyPatch, imported: _Imported
) -> tuple[MagicMock, list[str]]:
    loader = MagicMock(spec=GraphLoader)
    writers: list[str] = []

    def project_loader(writer: str) -> MagicMock:
        writers.append(writer)
        return loader

    catalog = _static_catalog(imported)
    monkeypatch.setattr("sema.cli_target._project_loader", project_loader)
    monkeypatch.setattr(
        "sema.cli_target._project_catalog_source", lambda *a, **k: catalog
    )
    return loader, writers


def _join_paths(loader: MagicMock) -> dict[str, dict[str, Any]]:
    paths = {}
    for call in loader.upsert_join_path.call_args_list:
        paths[call.args[0]] = call.kwargs
    return paths


def _uses(loader: MagicMock) -> dict[str, set[str]]:
    uses: dict[str, set[str]] = {}
    for call in loader.add_join_path_uses.call_args_list:
        key = call.args[1]
        uses.setdefault(call.args[0], set()).add(
            f"{key.catalog_or_db}.{key.schema}.{key.table}"
        )
    return uses


def _declared_pairs(imported: _Imported) -> list[tuple[str, str]]:
    return [
        (e["qualified_name"], join["referenced_entity"])
        for e in _table_rows(imported)
        for join in e.get("joins", [])
    ]


def _ref(table: str) -> str:
    return f"{_CATALOG}.{_SCHEMA}.{table}"


def test_target_project_writes_eight_declared_join_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, imported: _Imported
) -> None:
    loader, writers = _fake_project(monkeypatch, imported)
    tables = _tables(imported)

    result = CliRunner().invoke(
        target_group, _project_args(tmp_path, imported, tables)
    )

    assert result.exit_code == 0, result.output
    assert writers == ["in-memory"]
    paths = _join_paths(loader)
    assert len(paths) == len(FOREIGN_KEYS)
    assert {p["source"] for p in paths.values()} == {"manifest_declared"}
    assert {p["confidence"] for p in paths.values()} == {DECLARED_JOIN_CONFIDENCE}
    expected_pairs = {
        frozenset({_ref(tables[entity]), _ref(tables[referenced])})
        for entity, referenced in _declared_pairs(imported)
    }
    assert len(expected_pairs) == len(FOREIGN_KEYS)
    assert {frozenset(refs) for refs in _uses(loader).values()} == expected_pairs
    assert "paths written 8" in result.output
    assert "joins skipped 0" in result.output


def test_target_project_projects_every_bound_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, imported: _Imported
) -> None:
    loader, _ = _fake_project(monkeypatch, imported)
    tables = _tables(imported)

    result = CliRunner().invoke(
        target_group, _project_args(tmp_path, imported, tables)
    )

    assert result.exit_code == 0, result.output
    projected = {call.args[0] for call in loader.upsert_table.call_args_list}
    assert projected == set(tables.values())
    assert len(projected) == ENTITIES


def test_target_project_with_an_empty_binding_skips_all_eight_joins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, imported: _Imported
) -> None:
    loader, _ = _fake_project(monkeypatch, imported)

    result = CliRunner().invoke(target_group, _project_args(tmp_path, imported, {}))

    assert result.exit_code == 0, result.output
    loader.upsert_join_path.assert_not_called()
    assert "paths written 0" in result.output
    assert "joins skipped 8" in result.output


def test_target_project_without_binding_fails_naming_the_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, imported: _Imported
) -> None:
    loader, _ = _fake_project(monkeypatch, imported)

    result = CliRunner().invoke(target_group, _project_args(tmp_path, imported, None))

    assert result.exit_code != 0
    assert "--binding" in result.output
    loader.upsert_join_path.assert_not_called()


def test_target_project_in_memory_writer_never_opens_neo4j(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, imported: _Imported
) -> None:
    def no_driver(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the in-memory writer must not open a Neo4j driver")

    catalog = _static_catalog(imported)
    monkeypatch.setattr(
        "sema.cli_target._project_catalog_source", lambda *a, **k: catalog
    )
    monkeypatch.setattr("neo4j.GraphDatabase.driver", no_driver)

    result = CliRunner().invoke(
        target_group, _project_args(tmp_path, imported, _tables(imported))
    )

    assert result.exit_code == 0, result.output
    assert "paths written 8" in result.output
    assert "joins skipped 0" in result.output


def test_target_load_options_are_unchanged() -> None:
    from sema.cli_target import load_cmd

    assert [p.name for p in load_cmd.params] == ["manifest_path", "writer", "skip_facets"]

    result = CliRunner().invoke(
        target_group, ["load", "--manifest", str(_GOLDEN), "--binding", "b.yaml"]
    )

    assert result.exit_code != 0
    assert "No such option" in result.output
