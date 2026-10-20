"""CLI: `sema target load --manifest <path>` and `sema target import --spec <path>`.

Connects any user-supplied ontology manifest to the target loader and
prints a `LoadedTarget` summary as JSON. Defaults to the in-memory
writer for fast inspection; `--writer neo4j` materialises into the
graph using `Neo4jGraphWriter`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from sema.cli_target_utils import build_summary
from sema.connectors.base import Connector
from sema.targets.adapters.manifest import (
    ManifestTargetAdapter,
    register_manifest_adapter,
)
from sema.targets.loader import load_target
from sema.targets.materializer import GraphWriter, InMemoryGraphWriter

if TYPE_CHECKING:
    from sema.intake.relations import QueryFn


@click.group(name="target")
def target_group() -> None:
    """Target ontology operations."""
    register_manifest_adapter()


@target_group.command(name="load")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the YAML/JSON manifest declaring the target ontology.",
)
@click.option(
    "--writer",
    type=click.Choice(["in-memory", "neo4j"]),
    default="in-memory",
    show_default=True,
    help="Where to materialise the target. `in-memory` records ops without writing.",
)
@click.option(
    "--skip-facet",
    "skip_facets",
    multiple=True,
    help="Repeatable. Operator opt-out per facet (e.g. semantic_aliases).",
)
def load_cmd(manifest_path: Path, writer: str, skip_facets: tuple[str, ...]) -> None:
    """Load a manifest and print a LoadedTarget summary as JSON."""
    try:
        adapter = ManifestTargetAdapter(manifest_path)
        graph_writer = _build_writer(writer)
        loaded = load_target(
            adapter, writer=graph_writer, skip_facets=list(skip_facets)
        )
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(2)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo(json.dumps(build_summary(loaded), indent=2, default=str))


@target_group.command(name="import")
@click.option(
    "--spec",
    "spec_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Concept source spec (YAML) naming the library relations and column roles.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Where to write the target manifest.",
)
@click.option(
    "--evidence-out",
    "evidence_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Where to write the authored-evidence sidecar (JSON).",
)
@click.option(
    "--staging",
    "staging_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="DuckDB staging database for staging relations and term staging.",
)
def import_cmd(
    spec_path: Path, out_path: Path, evidence_path: Path, staging_path: Path | None
) -> None:
    """Import a concept library into a manifest plus an authored-evidence sidecar."""
    from sema.binding.importer import import_concept_source, write_import
    from sema.binding.spec import load_concept_source_spec, load_dataset_map
    from sema.ingest.duckdb_staging import Staging
    from sema.intake.relations import RelationProvider

    staging: Staging | None = None
    try:
        spec = load_concept_source_spec(spec_path)
        staging = Staging(str(staging_path), schemas=()) if staging_path else None
        bindings = spec.authored_bindings
        dataset_map = load_dataset_map(bindings.dataset_map) if bindings else {}
        provider = RelationProvider(warehouse=_lazy_warehouse(), staging=staging)
        result = import_concept_source(
            spec, provider, dataset_map=dataset_map, staging=staging
        )
        write_import(result, out_path, evidence_path)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        if staging is not None:
            staging.close()
    click.echo(f"Wrote {out_path} and {evidence_path}")


def _lazy_warehouse() -> QueryFn:
    """Builds the connector on the first warehouse query, never for staging specs."""
    connectors: list[Connector] = []

    def run(sql: str) -> list[tuple[Any, ...]]:
        if not connectors:
            connectors.append(_import_connector())
        return connectors[0].execute_query(sql)

    return run


def _import_connector() -> Connector:
    from sema.models.config import DatabricksConfig
    from sema.pipeline.build import DatabricksConnectorFactory

    # Same resolution as `sema build`: DATABRICKS_* environment settings.
    return DatabricksConnectorFactory(DatabricksConfig()).create()


def _build_writer(kind: str) -> GraphWriter:
    if kind == "in-memory":
        return InMemoryGraphWriter()
    if kind == "neo4j":
        return _neo4j_writer_from_env()
    raise click.BadParameter(f"unknown writer: {kind!r}")


def _neo4j_writer_from_env() -> GraphWriter:
    neo4j = __import__("neo4j")
    from sema.targets.neo4j_writer import Neo4jGraphWriter

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "graphrag")
    driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
    return Neo4jGraphWriter(driver)


__all__ = ["target_group", "load_cmd", "import_cmd"]
