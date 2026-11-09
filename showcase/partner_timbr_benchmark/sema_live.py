"""Live wiring for arm S on partner TEST: workspace, catalog, embedder, Neo4j.

Fails closed: the workspace must resolve to the TEST host, ambient
DATABRICKS_HOST/DATABRICKS_TOKEN are removed first, and no .env file is read.
Neo4j credentials come from NEO4J_USER/NEO4J_PASSWORD in the process env.
"""

from __future__ import annotations

import os
from typing import Any

from sema.graph.physical_catalog import PhysicalColumn

TEST_PROFILE = "TEST"
TEST_HOST = "https://partner-workspace.cloud.databricks.com"
EMBEDDER_ENDPOINT = "databricks-bge-large-en"


class WrongWorkspaceError(RuntimeError):
    """The resolved Databricks host is not the pinned TEST workspace."""


def prt_test_workspace() -> Any:
    from databricks.sdk import WorkspaceClient

    for key in ("DATABRICKS_HOST", "DATABRICKS_TOKEN"):
        os.environ.pop(key, None)
    os.environ["DATABRICKS_CONFIG_PROFILE"] = TEST_PROFILE
    workspace = WorkspaceClient(profile=TEST_PROFILE)
    assert_test_host(str(workspace.config.host))
    return workspace


def assert_test_host(host: str) -> None:
    if host.rstrip("/") != TEST_HOST:
        raise WrongWorkspaceError(f"resolved host {host!r} is not {TEST_HOST}")


class UnityCatalogSource:
    """`PhysicalCatalogSource` over Unity Catalog table metadata (no SQL)."""

    def __init__(self, workspace: Any) -> None:
        self._workspace = workspace

    def columns(self, *, catalog: str, schema: str, table: str) -> list[PhysicalColumn]:
        info = self._workspace.tables.get(f"{catalog}.{schema}.{table}")
        return [
            PhysicalColumn(str(c.name), str(c.type_text), bool(c.nullable))
            for c in info.columns or []
        ]


def embedder(workspace: Any) -> Any:
    from databricks_langchain import DatabricksEmbeddings

    assert_test_host(str(workspace.config.host))
    return DatabricksEmbeddings(
        endpoint=EMBEDDER_ENDPOINT, target_uri=f"databricks://{TEST_PROFILE}"
    )


def neo4j_driver(uri: str) -> Any:
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        uri, auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
    )
