"""Done-when 3: the neutral fixture, built through the DuckDB connector, scored from its graph.

The build runs with the model mocked and embeddings skipped, over the fixture's main schema.
`facts_from_graph(driver)` reads the graph's facts and `declared_constraints(connector,
catalog, schemas)` reads the constraints the DuckDB connector declares, each as `Cells` per
fact kind keyed as the oracle file format keys them. The fixture's shape: the graph holds
every declared key and every detected join, twelve joins in all, and contradicts no
declared key; the connector declares six keys.
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
    DETECTED_JOIN_KEYS,
    card_rows,
)

pytestmark = pytest.mark.integration

GRAPH_JOIN_KEYS = DECLARED_JOIN_KEYS | DETECTED_JOIN_KEYS


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


def test_the_graph_adapter_reads_every_join_the_build_wrote(built: Any) -> None:
    produced = _scorecard().facts_from_graph(built)

    assert set(produced["join"]) == GRAPH_JOIN_KEYS


def test_the_declared_constraint_adapter_reads_every_declared_key(
    connector: DuckDBConnector,
) -> None:
    oracle = _scorecard().declared_constraints(connector, CATALOG, [MAIN_SCHEMA])

    assert set(oracle["join"]) == DECLARED_JOIN_KEYS


def test_the_fixture_scored_from_its_graph_matches_its_shape(
    built: Any, connector: DuckDBConnector,
) -> None:
    scorecard = _scorecard()
    produced = scorecard.facts_from_graph(built)
    oracle = scorecard.declared_constraints(connector, CATALOG, [MAIN_SCHEMA])

    rows = card_rows(scorecard.render(scorecard.score_discovery(produced, oracle, {})))

    assert tuple(rows) == tuple(scorecard.FACT_KINDS)
    join = rows["join"]
    assert (join["produced"], join["wrong"]) == (str(len(GRAPH_JOIN_KEYS)), "0")
    assert int(join["equal"]) + int(join["missing"]) == len(DECLARED_JOIN_KEYS)
    assert int(join["extra"]) <= len(GRAPH_JOIN_KEYS - DECLARED_JOIN_KEYS)
