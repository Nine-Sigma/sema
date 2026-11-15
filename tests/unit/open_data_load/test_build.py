"""The build command reads its file, catalog and schema from a profile's `build:` keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from showcase.open_data_load.build import BuildTarget, build_config, read_build_target

pytestmark = pytest.mark.unit


@pytest.fixture
def profile(tmp_path: Path) -> Path:
    path = tmp_path / "p.yaml"
    path.write_text("build:\n  database: ~/x/widgets.duckdb\n  catalog: widgets\n  schema: parts\n")
    return path


def test_the_build_target_expands_home(profile: Path) -> None:
    assert read_build_target(profile) == BuildTarget(
        database=Path.home() / "x" / "widgets.duckdb", catalog="widgets", schema="parts",
    )


def test_the_build_config_reads_that_file_through_the_duckdb_source(profile: Path) -> None:
    config = build_config(read_build_target(profile), skip_embeddings=True)
    assert (config.source, config.catalog, config.schemas, config.skip_embeddings) == (
        "duckdb", "widgets", ["parts"], True,
    )
    assert config.duckdb.path == str(Path.home() / "x" / "widgets.duckdb")
