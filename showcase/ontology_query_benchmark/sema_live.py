"""Live wiring for arm S on the profile's pinned workspace: catalog, embedder, Neo4j.

Fails closed: the workspace must resolve to the pinned host, ambient
DATABRICKS_HOST/DATABRICKS_TOKEN are removed first, and no .env file is read.
Neo4j credentials come from NEO4J_USER/NEO4J_PASSWORD in the process env.
"""

from __future__ import annotations

import os
from typing import Any

from sema.graph.physical_catalog import PhysicalColumn
from showcase.ontology_query_benchmark.profile import Routing
from showcase.ontology_query_benchmark.routing import require_embedding_endpoint, require_host


def pinned_workspace(routing: Routing) -> Any:
    from databricks.sdk import WorkspaceClient

    for key in ("DATABRICKS_HOST", "DATABRICKS_TOKEN"):
        os.environ.pop(key, None)
    os.environ["DATABRICKS_CONFIG_PROFILE"] = routing.databricks_profile
    workspace = WorkspaceClient(profile=routing.databricks_profile)
    require_host(str(workspace.config.host), routing)
    return workspace


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


def embedder(workspace: Any, routing: Routing) -> Any:
    from databricks_langchain import DatabricksEmbeddings
    from mlflow.utils.databricks_utils import get_databricks_host_creds

    require_host(str(workspace.config.host), routing)
    model = DatabricksEmbeddings(
        endpoint=require_embedding_endpoint(routing.embedding_endpoint, routing),
        target_uri=f"databricks://{routing.databricks_profile}",
    )
    creds = get_databricks_host_creds(model.target_uri)  # type: ignore[no-untyped-call]
    require_host(str(creds.host), routing)
    return model


def neo4j_driver(uri: str) -> Any:
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        uri, auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
    )
