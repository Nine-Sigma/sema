"""CLI: `sema target load`, `sema target import`, and `sema target project`.

The intake commands `discover`, `approve`, and `runs` live in `sema.cli_intake` and are
registered on this group.

`load` connects any user-supplied ontology manifest to the target loader and
prints a `LoadedTarget` summary as JSON. Defaults to the in-memory
writer for fast inspection; `--writer neo4j` materialises into the
graph using `Neo4jGraphWriter`. `project` binds the manifest's entities to
physical tables and writes the retrieval projection plus one join path per
authored join.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from sema.cli_intake import approve_cmd, discover_cmd, runs_cmd
from sema.cli_target_utils import (
    RecordingDriver,
    build_summary,
    load_project_binding,
    warehouse_connector,
)
from sema.connectors.base import Connector
from sema.graph.loader import GraphLoader
from sema.graph.physical_catalog import DatabricksCatalogSource, PhysicalCatalogSource
from sema.graph.target_join_paths import materialize_target_join_paths
from sema.graph.target_retrieval_loader import project_target_to_retrieval_graph
from sema.targets.adapters.manifest import (
    ManifestTargetAdapter,
    register_manifest_adapter,
)
from sema.targets.loader import load_target
from sema.targets.materializer import GraphWriter, InMemoryGraphWriter
from sema.targets.normalizer import TargetModelNormalizer

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


@target_group.command(name="project")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the YAML/JSON manifest declaring the target ontology.",
)
@click.option(
    "--binding",
    "binding_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="YAML naming catalog, schema, and the physical table of each entity.",
)
@click.option(
    "--writer",
    type=click.Choice(["in-memory", "neo4j"]),
    default="in-memory",
    show_default=True,
    help="Where to write the projection. `in-memory` records statements only.",
)
def project_cmd(manifest_path: Path, binding_path: Path, writer: str) -> None:
    """Project bound target tables and their authored joins into the graph."""
    try:
        normalized = TargetModelNormalizer.normalize(ManifestTargetAdapter(manifest_path))
        binding = load_project_binding(binding_path)
        catalog = _project_catalog_source(binding.catalog)
        loader = _project_loader(writer)
        try:
            project_target_to_retrieval_graph(loader, normalized, binding, catalog)
            counts = materialize_target_join_paths(loader, normalized, binding)
        finally:
            _close_loader(loader)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo(f"paths written {counts.written}")
    click.echo(f"joins skipped {counts.skipped}")


def _project_catalog_source(catalog: str) -> PhysicalCatalogSource:
    from sema.cli_fit_utils import open_databricks_cursor
    from sema.models.config import DatabricksConfig

    return DatabricksCatalogSource(
        open_databricks_cursor(DatabricksConfig(), catalog=catalog)
    )


def _project_loader(writer: str) -> GraphLoader:
    if writer == "in-memory":
        return GraphLoader(RecordingDriver())
    if writer == "neo4j":
        from sema.cli_factories import _get_neo4j_driver
        from sema.models.config import Neo4jConfig

        return GraphLoader(_get_neo4j_driver(Neo4jConfig()))
    raise click.BadParameter(f"unknown writer: {writer!r}")


def _close_loader(loader: GraphLoader) -> None:
    # GraphLoader does not own its driver's lifecycle; the command that built it does.
    driver = getattr(loader, "_driver", None)
    if driver is not None:
        driver.close()


def _lazy_warehouse() -> QueryFn:
    """Builds the connector on the first warehouse query, never for staging specs."""
    connectors: list[Connector] = []

    def run(sql: str) -> list[tuple[Any, ...]]:
        if not connectors:
            connectors.append(_import_connector())
        return connectors[0].execute_query(sql)

    return run


def _import_connector() -> Connector:
    return warehouse_connector()


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


target_group.add_command(discover_cmd)
target_group.add_command(approve_cmd)
target_group.add_command(runs_cmd)

__all__ = ["target_group", "load_cmd", "import_cmd", "project_cmd"]
