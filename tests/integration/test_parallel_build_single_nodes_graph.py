"""bug-1153: a build writes one Catalog, Schema and Vocabulary node per key under parallel workers.

The neutral fixture is built twice, each time on an empty graph with one table worker per table,
so every worker writes the shared catalog, its schema and the vocabularies it detects at once.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from sema.models.config import Neo4jConfig
from sema.pipeline.orchestrate import run_build
from tests.support.connector_fixture import (
    CATALOG,
    SCHEMA_TABLES,
    fake_model_environment,
    fixture_build_config,
    write_fixture_database,
)

pytestmark = pytest.mark.integration

Key = tuple[str, tuple[Any, ...]]

_NODES_PER_KEY = (
    "MATCH (c:Catalog) RETURN 'Catalog' AS label, [c.name] AS key, count(*) AS nodes "
    "UNION ALL "
    "MATCH (s:Schema) RETURN 'Schema' AS label, [s.catalog, s.name] AS key, count(*) AS nodes "
    "UNION ALL "
    "MATCH (v:Vocabulary) RETURN 'Vocabulary' AS label, [v.name] AS key, count(*) AS nodes"
)


def _neo4j_config() -> Neo4jConfig:
    return Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=SecretStr(os.getenv("NEO4J_PASSWORD", "graphrag")),
    )


def _parallel_build(driver: Any, database: Path) -> dict[Key, int]:
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n").consume()
    workers = sum(len(tables) for tables in SCHEMA_TABLES.values())
    config = fixture_build_config(_neo4j_config(), database=database).model_copy(
        update={"table_workers": workers},
    )
    with fake_model_environment():
        report = run_build(config)
    assert report["tables_failed"] == 0, report["failed_tables"]
    with driver.session() as session:
        return {
            (row["label"], tuple(row["key"])): row["nodes"]
            for row in session.run(_NODES_PER_KEY)
        }


def _keys(nodes_per_key: dict[Key, int], label: str) -> set[tuple[Any, ...]]:
    return {key for (found, key) in nodes_per_key if found == label}


def test_each_parallel_build_holds_one_node_per_catalog_schema_and_vocabulary_key(
    clean_neo4j: Any, tmp_path: Path,
) -> None:
    database = write_fixture_database(tmp_path)
    duplicates: dict[int, dict[Key, int]] = {}
    for build in (1, 2):
        nodes_per_key = _parallel_build(clean_neo4j, database)
        assert _keys(nodes_per_key, "Catalog") == {(CATALOG,)}, (build, nodes_per_key)
        assert _keys(nodes_per_key, "Schema") == {
            (CATALOG, schema) for schema in SCHEMA_TABLES
        }, (build, nodes_per_key)
        assert _keys(nodes_per_key, "Vocabulary"), (build, "the build wrote no Vocabulary")
        duplicates[build] = {key: n for key, n in nodes_per_key.items() if n != 1}
    assert duplicates == {1: {}, 2: {}}, duplicates
