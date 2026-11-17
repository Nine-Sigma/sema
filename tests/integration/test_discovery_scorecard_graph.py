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
import json
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from pydantic import SecretStr

from sema.connectors.duckdb import DuckDBConnector
from sema.graph.join_materializer import materialize_join_paths
from sema.graph.loader import GraphLoader
from sema.graph.target_join_paths import materialize_target_join_paths
from sema.graph.target_retrieval_loader import EntityTableBinding, PhysicalSchemaBinding
from sema.models.assertions import Assertion, AssertionPredicate, AssertionStatus
from sema.models.config import DuckDBConfig, Neo4jConfig
from sema.models.planner._enums import TargetArtifactKind
from sema.models.target.completeness import (
    SemanticCompleteness,
    SemanticCompletenessAnnotations,
)
from sema.models.target.declared import DeclaredJoinDecl
from sema.models.target.descriptor import TargetModelDescriptor
from sema.models.target.entity import TargetEntityDecl
from sema.models.target.normalized import NormalizedTargetModel
from sema.models.target.refs import TargetEntityRef
from sema.pipeline.orchestrate import run_build
from tests.support.connector_fixture import (
    CATALOG,
    DECOY_SCHEMA,
    MAIN_SCHEMA,
    MAIN_TABLES,
    fake_model_environment,
    fixture_build_config,
    write_fixture_database,
)
from tests.support.connector_fixture_scorecard import (
    DECLARED_JOIN_KEYS,
    DECLARED_JOIN_VALUES,
    DETECTED_JOIN_KEYS,
    card_rows,
    join_key,
)

pytestmark = pytest.mark.integration

GRAPH_JOIN_KEYS = DECLARED_JOIN_KEYS | DETECTED_JOIN_KEYS
SINGLE_COLUMN_GRAPH_JOIN_KEYS = frozenset(k for k in GRAPH_JOIN_KEYS if "+" not in k[1])
OTHER_CATALOG = "othercat"
DECOY_SCHEMA_JOIN = join_key("ledger", ["holder_id"], "holder", ["holder_id"])
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


def _ref(catalog: str, schema: str, table: str) -> str:
    return f"databricks://ds/{catalog}/{schema}/{table}"


def _build_join_path(loader: GraphLoader, catalog: str, schema: str, table: str) -> None:
    """A join path the build's writer stores for `schema`, over two tables it upserts there."""
    referenced, referencing = ("holder", "holder_id"), (table, "holder_id")
    for table, column in (referenced, referencing):
        loader.upsert_table(table, schema, catalog, ref=_ref(catalog, schema, table))
        loader.upsert_column(column, table, schema, catalog, "INTEGER")
    predicate = {
        "left_table": referenced[0], "left_column": referenced[1],
        "right_table": referencing[0], "right_column": referencing[1], "operator": "=",
    }
    join = Assertion(
        id=f"{catalog}-{schema}", subject_ref=_ref(catalog, schema, referencing[0]),
        object_ref=_ref(catalog, schema, referenced[0]), predicate=AssertionPredicate.JOINS_TO,
        payload={
            "join_predicates": [predicate], "hop_count": 1,
            "from_ref": _ref(catalog, schema, referenced[0]),
            "to_ref": _ref(catalog, schema, referencing[0]),
        },
        source="connector", confidence=0.95, status=AssertionStatus.AUTO, run_id="run",
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    materialize_join_paths(loader, {("s", "joins_to"): [join]}, source_schema=schema)


def _target_join_path(loader: GraphLoader) -> None:
    """A target-loader join path over two of the fixture's tables, in the configured schema."""
    def entity(name: str, joins: list[DeclaredJoinDecl]) -> TargetEntityDecl:
        ref = TargetEntityRef(
            target_model_id="m", qualified_name=name, kind=TargetArtifactKind.TABLE_ROW,
        )
        return TargetEntityDecl(ref=ref, declared_joins=joins)

    join = DeclaredJoinDecl(
        referenced_entity="ns.referenced", join_keys=[("product_id", "product_id")],
        evidence="declared",
    )
    none = SemanticCompleteness.NONE
    model = NormalizedTargetModel(
        descriptor=TargetModelDescriptor(
            target_model_id="m", target_model_version="1", display_name="M",
            completeness=SemanticCompletenessAnnotations(
                structure=SemanticCompleteness.COMPLETE, obligations=none,
                vocabulary_bindings=none, semantic_aliases=none, terms=none,
            ),
        ),
        entities=[entity("ns.referenced", []), entity("ns.referencing", [join])],
    )
    binding = PhysicalSchemaBinding(catalog=CATALOG, schema=MAIN_SCHEMA, entity_tables=(
        EntityTableBinding("ns.referenced", "product"),
        EntityTableBinding("ns.referencing", "price_period"),
    ))
    assert materialize_target_join_paths(loader, model, binding).written == 1


@pytest.fixture
def foreign_joins(built: Any) -> Any:
    """The built graph plus a build join path in a second schema and in a second catalog, and
    a target-loader join path in the configured schema."""
    loader = GraphLoader(built)
    _build_join_path(loader, CATALOG, DECOY_SCHEMA, DECOY_SCHEMA_JOIN[0])
    _build_join_path(loader, OTHER_CATALOG, MAIN_SCHEMA, "journal")
    _target_join_path(loader)
    return built


def _join_path_count(driver: Any) -> int:
    with driver.session() as session:
        return int(session.run("MATCH (jp:JoinPath) RETURN count(jp) AS n").single()["n"])


def test_the_graph_adapter_reads_only_the_configured_schemas_build_join_paths(
    foreign_joins: Any,
) -> None:
    produced = _scorecard().facts_from_graph(foreign_joins, CATALOG, [MAIN_SCHEMA])

    assert _join_path_count(foreign_joins) == len(GRAPH_JOIN_KEYS) + 3
    assert set(produced["join"]) == GRAPH_JOIN_KEYS


def test_with_no_schemas_the_catalog_bounds_the_graph_read(foreign_joins: Any) -> None:
    produced = _scorecard().facts_from_graph(foreign_joins, CATALOG, None)

    assert set(produced["join"]) == GRAPH_JOIN_KEYS | {DECOY_SCHEMA_JOIN}


def _config_file(directory: Path, database: Path) -> Path:
    neo4j = _neo4j_config()
    path = directory / "build.yaml"
    path.write_text(yaml.safe_dump({
        "source": "duckdb", "catalog": CATALOG, "schemas": [MAIN_SCHEMA],
        "duckdb": {"path": str(database)},
        "neo4j": {"uri": neo4j.uri, "user": neo4j.user,
                  "password": neo4j.password.get_secret_value()},
    }))
    return path


def _discovery(tmp_path: Path, database: Path, *oracle: str) -> dict[str, dict[str, str]]:
    from sema.cli_eval import eval_group

    out = tmp_path / "card.md"
    config = _config_file(tmp_path, database)
    result = CliRunner().invoke(
        eval_group, ["discovery", "--config", str(config), *oracle, "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    rows = card_rows(out.read_text())
    assert tuple(rows) == tuple(_scorecard().FACT_KINDS)
    return rows


def test_the_discovery_command_scores_the_graph_against_the_declared_constraints(
    foreign_joins: Any, database: Path, tmp_path: Path,
) -> None:
    rows = _discovery(tmp_path, database)

    assert {cell: rows["join"][cell] for cell in JOIN_SHAPE} == {
        cell: str(count) for cell, count in JOIN_SHAPE.items()
    }
    assert rows["join"]["rechecked"] == str(len(SINGLE_COLUMN_GRAPH_JOIN_KEYS))
    assert rows["candidate_key"]["missing"] == str(len(MAIN_TABLES))


def test_the_discovery_command_scores_the_graph_against_an_oracle_file(
    foreign_joins: Any, database: Path, tmp_path: Path,
) -> None:
    oracle = tmp_path / "oracle.jsonl"
    oracle.write_text("".join(
        json.dumps({"kind": "join", "key": list(key), "value": value,
                    "basis": "data", "cite": "fixture"}) + "\n"
        for key, value in DECLARED_JOIN_VALUES.items()
    ))

    rows = _discovery(tmp_path, database, "--oracle-spec", str(oracle))

    assert {cell: rows["join"][cell] for cell in JOIN_SHAPE} == {
        cell: str(count) for cell, count in JOIN_SHAPE.items()
    }
    assert "no oracle" in rows["candidate_key"].values()
