"""Done-when 1, the second review's cases: placement by equality only, the stronger of two
contained candidates, a name-matched code relation's labels, and the sibling violation floor.

The build reads the coded fixture plus the case relations and the dictionary plus the case
entry; expected numbers come from the fixture's rows.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from neo4j import GraphDatabase
from pydantic import SecretStr

from sema.models.config import BuildConfig, DuckDBConfig, Neo4jConfig
from sema.pipeline.orchestrate import run_build
from tests.integration.test_coded_columns_graph import (
    Graph,
    _members,
    _one_verified,
    _read,
)
from tests.support.connector_fixture import fake_model_environment
from tests.support.connector_fixture_codes import (
    CASE_ENTRIES,
    CASE_TABLES,
    CODED_TABLES,
    CODES_CATALOG,
    CODES_SCHEMA,
    DICTIONARY_ENTRIES,
    DYE,
    GLAZE,
    GLINT,
    HUE,
    HUE_NOISE_PERIOD,
    STRIP_ROWS,
    column_values,
    write_coded_database,
    write_dictionary,
)

pytestmark = pytest.mark.integration


def _neo4j() -> Neo4jConfig:
    return Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=SecretStr(os.getenv("NEO4J_PASSWORD", "graphrag")),
    )


def _build(graph: Graph, directory: Path) -> None:
    with graph.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    database = write_coded_database(directory, (*CODED_TABLES, *CASE_TABLES))
    dictionary = write_dictionary(directory, (*DICTIONARY_ENTRIES, *CASE_ENTRIES))
    config = BuildConfig.model_validate({
        "source": "duckdb", "catalog": CODES_CATALOG, "schemas": [CODES_SCHEMA],
        "duckdb": DuckDBConfig(path=str(database)), "skip_embeddings": True,
        "table_workers": 1, "neo4j": _neo4j(),
        "concept_spec": str(dictionary.spec), "concept_staging": str(dictionary.staging),
    })
    with fake_model_environment():
        report = run_build(config)
    if report["tables_failed"]:
        graph.error = f"tables failed: {report['failed_tables']}"
        return
    _read(graph)


@pytest.fixture(scope="module")
def graph(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Graph]:
    neo4j = _neo4j()
    driver = GraphDatabase.driver(neo4j.uri, auth=(neo4j.user, neo4j.password.get_secret_value()))
    built = Graph(driver)
    try:
        _build(built, tmp_path_factory.mktemp("coded_cases"))
    except Exception as exc:  # a build crash fails every test with its cause, never an error
        built.error = f"{type(exc).__name__}: {exc}"
    yield built
    driver.close()


@pytest.mark.parametrize("prop", sorted(CASE_ENTRIES[0].domains))
def test_a_property_equal_to_no_column_name_stays_unplaced(graph: Graph, prop: str) -> None:
    stated = [e for found in graph.built().evidence.values() for e in found
              if e.kind == "stated_domain"]
    unplaced = [e for e in stated if e.payload.get("property") == prop]
    assert unplaced and all(e.payload.get("reason") for e in unplaced), unplaced
    placed_columns = {column for column, ref in graph.refs.items()
                      if any(e.kind == "stated_domain" for e in graph.evidence.get(ref, []))}
    assert ("panel", "coat_kind") in placed_columns
    assert not any(e.payload.get("property") == prop
                   for column in placed_columns for e in graph.evidence_of(column))


def test_the_stronger_of_two_contained_candidates_verifies_and_the_other_is_recorded(
    graph: Graph,
) -> None:
    column = ("strip", "glint_code")
    link = _one_verified(graph.built(), column, GLINT)
    assert link.members == _members(column, dict(GLINT))
    full = {e.object_ref for e in graph.containment(column)
            if e.payload.get("listed_value_share") == 1.0}
    assert {graph.refs[("glint", "glint_code")], graph.refs[("spark", "spark_code")]} <= full
    assert graph.objects(column, "name_match") == {graph.refs[("glint", "glint_code")]}


def test_a_name_matched_code_relation_labels_its_members_by_its_text_column(
    graph: Graph,
) -> None:
    column = ("fitting", "glaze_code")
    link = _one_verified(graph.built(), column, GLAZE)
    assert link.members == _members(column, dict(GLAZE))
    ((label_column, props),) = graph.verified_labels(column)
    assert label_column == ("glaze", "glaze_title")
    assert props.get("violation_rate") == 0.0
    via = props.get("via") or []
    assert via and all({"fitting", "glaze"} <= graph.join_paths.get(name, set()) for name in via)


def test_a_sibling_within_the_floor_labels_with_its_violation_rate(graph: Graph) -> None:
    column = ("strip", "hue_code")
    link = _one_verified(graph.built(), column, HUE)
    assert link.members == _members(column, dict(HUE))
    ((label_column, props),) = graph.verified_labels(column)
    assert label_column == ("strip", "hue_name")
    noisy = sum(1 for i in range(STRIP_ROWS) if i % HUE_NOISE_PERIOD == 0)
    assert props.get("violation_rate") == pytest.approx(noisy / STRIP_ROWS)
    assert not props.get("via")


def test_a_sibling_past_the_floor_is_no_label(graph: Graph) -> None:
    column = ("strip", "dye_code")
    link = _one_verified(graph.built(), column, DYE)
    assert link.members == _members(column, dict.fromkeys(DYE))
    assert graph.verified_labels(column) == []
    assert len(set(column_values("strip", "dye_name"))) == len(DYE)
