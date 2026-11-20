"""Build the graph of a DuckDB file a profile names, then report its joins as JSON.

Run as `python -m showcase.open_data_load.build --profile <yaml> [--skip-embeddings]`.
The profile's `build:` keys name the file, its catalog and the schema to build; Neo4j
comes from the `NEO4J_*` environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from sema.models.config import BuildConfig, DuckDBConfig
from sema.pipeline.orchestrate import run_build
from showcase.open_data_load.build_utils import join_report


@dataclass(frozen=True)
class BuildTarget:
    database: Path
    catalog: str
    schema: str


def read_build_target(profile_path: Path) -> BuildTarget:
    keys = yaml.safe_load(profile_path.read_text(encoding="utf-8"))["build"]
    return BuildTarget(
        database=Path(keys["database"]).expanduser(),
        catalog=keys["catalog"],
        schema=keys["schema"],
    )


def build_config(target: BuildTarget, *, skip_embeddings: bool) -> BuildConfig:
    return BuildConfig(
        source="duckdb",
        catalog=target.catalog,
        schemas=[target.schema],
        duckdb=DuckDBConfig(path=str(target.database)),
        skip_embeddings=skip_embeddings,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args(argv)
    target = read_build_target(args.profile)
    config = build_config(target, skip_embeddings=args.skip_embeddings)
    build = run_build(config)
    report = join_report(config, target.database, build)
    print(json.dumps(report, default=str))
    return 0 if build.get("tables_failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
