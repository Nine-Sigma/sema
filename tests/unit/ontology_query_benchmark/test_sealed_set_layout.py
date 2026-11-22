"""Arm A's schema on a DuckDB file whose tables live in DuckDB's default schema `main`."""

import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest
import yaml

from tests.support.connector_fixture import (
    CATALOG,
    MAIN_SCHEMA,
    MAIN_TABLES,
    write_fixture_database,
)
from tests.unit.ontology_query_benchmark.sealed_set_env import materialize, run_arms

pytestmark = pytest.mark.unit

DEFAULT_SCHEMA = "main"


def _point_profile_at_default_schema_copy(root: Path, profile_path: Path) -> list[str]:
    """Copy the fixture's tables into `main` of a new file named as the catalog; return
    their relations."""
    (root / "fixture_source").mkdir()
    (root / "default_schema").mkdir()
    source = write_fixture_database(root / "fixture_source")
    target = root / "default_schema" / f"{CATALOG}.duckdb"
    con = duckdb.connect(str(target))
    try:
        con.execute(f"ATTACH '{source}' AS fixture_source (READ_ONLY)")
        for table in MAIN_TABLES:
            con.execute(f"CREATE TABLE {DEFAULT_SCHEMA}.{table.name} AS "
                        f"SELECT * FROM fixture_source.{MAIN_SCHEMA}.{table.name}")  # fmt: skip
    finally:
        con.close()
    raw = yaml.safe_load(profile_path.read_text())
    raw["duckdb"] = str(target)
    profile_path.write_text(yaml.safe_dump(raw))
    return [f"{CATALOG}.{DEFAULT_SCHEMA}.{table.name}" for table in MAIN_TABLES]


def test_tables_in_the_default_schema_reach_the_schema_cache_and_arm_a_prompt(
    frozen: Path, harness: SimpleNamespace, tmp_path: Path,
) -> None:
    root = tmp_path / "sealed_set"
    profile_path = materialize(root, frozen)
    relations = _point_profile_at_default_schema_copy(tmp_path, profile_path)

    run = run_arms(harness, root, profile_path, ("A",))

    cache = run.set_dir / "sealed" / "runs" / "Ax1" / "cache" / "schema.json"
    assert sorted(json.loads(cache.read_text())) == sorted(relations)
    first_prompt = "\n".join(str(m["content"]) for m in run.model.calls[0])
    assert [r for r in relations if r not in first_prompt] == []
