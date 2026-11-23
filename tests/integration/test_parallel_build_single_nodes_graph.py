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
    MAIN_SCHEMA,
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


def test_a_build_keeps_each_edge_of_a_duplicate_schema_it_does_not_rebuild(
    clean_neo4j: Any, tmp_path: Path,
) -> None:
    other = "other_study"
    with clean_neo4j.session() as session:
        session.run("DROP CONSTRAINT schema_catalog_name_unique IF EXISTS").consume()
        session.run(
            "UNWIND ['t1', 't2'] AS table "
            "CREATE (:Table {name: table, schema_name: $other, catalog: $catalog})"
            "-[:IN_SCHEMA]->(:Schema {name: $other, catalog: $catalog})",
            other=other, catalog=CATALOG,
        ).consume()
    config = fixture_build_config(_neo4j_config(), database=write_fixture_database(tmp_path))
    with fake_model_environment():
        report = run_build(config)
    assert report["tables_failed"] == 0, report["failed_tables"]
    with clean_neo4j.session() as session:
        row = session.run(
            "MATCH (s:Schema {name: $other, catalog: $catalog}) "
            "OPTIONAL MATCH (t:Table)-[:IN_SCHEMA]->(s) "
            "RETURN count(DISTINCT s) AS schemas, collect(t.name) AS tables",
            other=other, catalog=CATALOG,
        ).single()
    assert (row["schemas"], sorted(row["tables"])) == (1, ["t1", "t2"])


def test_a_schema_name_under_a_second_catalog_stays_its_own_node_with_its_edges(
    clean_neo4j: Any, tmp_path: Path,
) -> None:
    second = "second_catalog"
    tables = sorted(table.name for table in SCHEMA_TABLES[MAIN_SCHEMA])
    with clean_neo4j.session() as session:
        for constraint in ("schema_catalog_name_unique", "catalog_name_unique"):
            session.run(f"DROP CONSTRAINT {constraint} IF EXISTS").consume()
        session.run(
            "UNWIND [$catalog, $catalog, $second] AS catalog "
            "CREATE (s:Schema {name: $schema, catalog: catalog})"
            "-[:IN_CATALOG]->(:Catalog {name: catalog}) "
            "WITH s, catalog WHERE catalog = $second "
            "UNWIND $tables AS table "
            "CREATE (:Table {name: table, schema_name: $schema, catalog: catalog})"
            "-[:IN_SCHEMA]->(s)",
            catalog=CATALOG, second=second, schema=MAIN_SCHEMA, tables=tables,
        ).consume()
    config = fixture_build_config(_neo4j_config(), database=write_fixture_database(tmp_path))
    with fake_model_environment():
        report = run_build(config)
    assert report["tables_failed"] == 0, report["failed_tables"]
    with clean_neo4j.session() as session:
        catalogs = {
            row["name"]: row["nodes"]
            for row in session.run("MATCH (c:Catalog) RETURN c.name AS name, count(*) AS nodes")
        }
        schemas = {
            row["catalog"]: (row["schemas"], sorted(row["tables"]), row["catalogs"])
            for row in session.run(
                "MATCH (s:Schema {name: $schema}) "
                "OPTIONAL MATCH (t:Table)-[:IN_SCHEMA]->(s) "
                "OPTIONAL MATCH (s)-[:IN_CATALOG]->(c:Catalog) "
                "RETURN s.catalog AS catalog, count(DISTINCT s) AS schemas, "
                "collect(DISTINCT t.name) AS tables, collect(DISTINCT c.name) AS catalogs",
                schema=MAIN_SCHEMA,
            )
        }
    assert catalogs == {CATALOG: 1, second: 1}, catalogs
    assert schemas == {
        CATALOG: (1, tables, [CATALOG]),
        second: (1, tables, [second]),
    }, schemas
