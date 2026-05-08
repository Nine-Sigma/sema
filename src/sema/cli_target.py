"""CLI: `sema target load --manifest <path>`.

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

import click

from sema.cli_target_utils import build_summary
from sema.targets.adapters.manifest import (
    ManifestTargetAdapter,
    register_manifest_adapter,
)
from sema.targets.loader import load_target
from sema.targets.materializer import GraphWriter, InMemoryGraphWriter


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
def load_cmd(
    manifest_path: Path, writer: str, skip_facets: tuple[str, ...]
) -> None:
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


__all__ = ["target_group", "load_cmd"]
