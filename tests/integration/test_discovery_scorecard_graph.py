"""Done-when 3: the neutral fixture, built through the DuckDB connector, scored from its graph.

The build runs with the model mocked and embeddings skipped, over the fixture's main schema.
`facts_from_graph(driver)` reads the graph's facts and `declared_constraints(connector,
catalog, schemas)` reads the constraints the DuckDB connector declares, each as `Cells` per
fact kind keyed as the oracle file format keys them. The fixture's shape: the graph holds
every declared key and every detected join, twelve joins in all, none with a cardinality; the
connector declares six keys, none on columns declared unique, so each is `many_to_one`.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from sema.connectors.duckdb import DuckDBConnector
from sema.models.config import DuckDBConfig, Neo4jConfig
from sema.pipeline.orchestrate import run_build
from tests.support.connector_fixture import (
    CATALOG,
    MAIN_SCHEMA,
    fake_model_environment,
    fixture_build_config,
    write_fixture_database,
)
from tests.support.connector_fixture_scorecard import (
    DECLARED_JOIN_KEYS,
    DECLARED_JOIN_VALUES,
    DETECTED_JOIN_KEYS,
    card_rows,
)

pytestmark = pytest.mark.integration

GRAPH_JOIN_KEYS = DECLARED_JOIN_KEYS | DETECTED_JOIN_KEYS
KEY_ONLY = "key_only"
# The graph records no cardinality, so every produced join is key_only, and key_only is equal
# to an oracle claim on the same key whatever the oracle's cardinality.
JOIN_SHAPE = {"produced": 12, "equal": 6, "missing": 0, "wrong": 0, "extra": 6}


def _scorecard() -> Any:
    return importlib.import_module("sema.eval.discovery_scorecard")


def _neo4j_config() -> Neo4jConfig:
    return Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=SecretStr(os.getenv("NEO4J_PASSWORD", "graphrag")),
    )


@pytest.fixture
def database(tmp_path: Path) -> Path:
    return write_fixture_database(tmp_path)


@pytest.fixture
def built(clean_neo4j: Any, database: Path) -> Any:
    config = fixture_build_config(_neo4j_config(), database=database).model_copy(
        update={"schemas": [MAIN_SCHEMA]},
    )
    with fake_model_environment():
        report = run_build(config)
    assert report["tables_failed"] == 0, report["failed_tables"]
    return clean_neo4j


@pytest.fixture
def connector(database: Path) -> Iterator[DuckDBConnector]:
    opened = DuckDBConnector(DuckDBConfig(path=str(database)))
    yield opened
    opened.close()


def _join_paths_without_cardinality(driver: Any) -> int:
    with driver.session() as session:
        record = session.run(
            "MATCH (jp:JoinPath) WHERE jp.cardinality_hint IS NULL RETURN count(jp) AS n"
        ).single()
    return int(record["n"])


def test_the_fixture_reading_behind_the_expected_counts(built: Any) -> None:
    assert (len(GRAPH_JOIN_KEYS), len(DECLARED_JOIN_KEYS & GRAPH_JOIN_KEYS)) == (12, 6)
    assert _join_paths_without_cardinality(built) == len(GRAPH_JOIN_KEYS)
    assert set(DECLARED_JOIN_VALUES.values()) == {"many_to_one"}


def test_the_graph_adapter_reads_every_join_the_build_wrote(built: Any) -> None:
    produced = _scorecard().facts_from_graph(built)

    assert set(produced["join"]) == GRAPH_JOIN_KEYS
    key_only = [key for key, value in produced["join"].items() if value == KEY_ONLY]
    assert len(key_only) == _join_paths_without_cardinality(built)


def test_the_declared_constraint_adapter_reads_every_declared_key(
    connector: DuckDBConnector,
) -> None:
    oracle = _scorecard().declared_constraints(connector, CATALOG, [MAIN_SCHEMA])

    assert oracle["join"] == DECLARED_JOIN_VALUES


def test_the_fixture_scored_from_its_graph_matches_its_shape(
    built: Any, connector: DuckDBConnector,
) -> None:
    scorecard = _scorecard()
    produced = scorecard.facts_from_graph(built)
    oracle = scorecard.declared_constraints(connector, CATALOG, [MAIN_SCHEMA])

    rows = card_rows(scorecard.render(scorecard.score_discovery(produced, oracle, {})))

    assert tuple(rows) == tuple(scorecard.FACT_KINDS)
    join = rows["join"]
    assert {cell: join[cell] for cell in JOIN_SHAPE} == {
        cell: str(count) for cell, count in JOIN_SHAPE.items()
    }
