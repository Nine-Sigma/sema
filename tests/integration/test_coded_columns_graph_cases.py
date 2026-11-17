"""Done-when 1, the second review's cases: placement by equality only, the stronger of two
contained candidates, a name-matched code relation's labels, and the sibling violation floor.
Then rulings E to G: a relation's own key is no coded column, a stated domain past the bound
is recorded, and placement runs once across the schemas of a run.

The build reads the coded fixture plus the case relations and the dictionary plus the case
entries; a second build reads two schemas. Expected numbers come from the fixture's rows.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from neo4j import GraphDatabase
from pydantic import SecretStr

from sema.models.config import BuildConfig, DuckDBConfig, Neo4jConfig
from sema.pipeline.orchestrate import run_build
from tests.integration.test_coded_columns_graph import (
    _COLUMNS,
    _LINKS,
    Evidence,
    Graph,
    _members,
    _one_verified,
    _read,
)
from tests.support.connector_fixture import (
    FixtureTable,
    fake_model_environment,
    write_fixture_database,
)
from tests.support.connector_fixture_codes import (
    ANNEX_SCHEMA,
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
    LOW_DISTINCT_BOUND,
    PAST_BOUND_PROPERTY,
    SEAL_KINDS,
    SPLIT_ENTRIES,
    SPLIT_PLACED_PROPERTY,
    SPLIT_SCHEMA_TABLES,
    SPLIT_UNPLACED_PROPERTY,
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


def _all_distinct(table: FixtureTable) -> list[tuple[str, str]]:
    present = {name: [v for v in column_values(table.name, name) if v is not None]
               for name in table.column_names}
    return [(table.name, name) for name, values in present.items()
            if values and len(set(values)) == len(values)]


def test_a_relations_own_key_is_never_a_coded_column(graph: Graph) -> None:
    rack, spare = ("rack", "rack_no"), ("rack_spare", "rack_no")
    full = {e.object_ref for e in graph.built().containment(("panel", "rack_no"))
            if e.payload.get("listed_value_share") == 1.0}
    assert {graph.refs[rack], graph.refs[spare]} <= full, "each rack key keeps its own codes"
    keys = [c for t in (*CODED_TABLES, *CASE_TABLES) for c in _all_distinct(t)]
    assert {rack, spare} <= set(keys)
    for column in keys:
        assert graph.links.get(column, []) == [], column
        assert graph.labeled_by.get(column, []) == [], column
        assert [e for e in graph.evidence_of(column) if e.predicate == "has_value_set"] == []
    assert not any({"rack", "rack_spare"} <= tables for tables in graph.join_paths.values())


def test_a_stated_domain_on_a_column_past_the_bound_is_recorded_with_the_reason(
    graph: Graph,
) -> None:
    column = ("panel", "assigned_crew")
    assert len({v for v in column_values(*column) if v is not None}) == LOW_DISTINCT_BOUND + 1
    stated = [e for e in graph.built().evidence_of(column) if e.kind == "stated_domain"]
    assert [e.payload.get("property") for e in stated] == [PAST_BOUND_PROPERTY]
    assert "past the low-cardinality bound" in stated[0].payload.get("reason", "")
    assert stated[0].payload.get("linked") is False
    assert graph.links.get(column, []) == []
    assert graph.containment(column) == []


@dataclass
class Split:
    works: Graph
    annex_refs: dict[tuple[str, str], str] = field(default_factory=dict)
    annex_links: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)


def _build_split(split: Split, directory: Path) -> None:
    with split.works.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    database = write_fixture_database(directory, SPLIT_SCHEMA_TABLES, CODES_CATALOG)
    dictionary = write_dictionary(directory, SPLIT_ENTRIES)
    config = BuildConfig.model_validate({
        "source": "duckdb", "catalog": CODES_CATALOG,
        "schemas": sorted(SPLIT_SCHEMA_TABLES), "duckdb": DuckDBConfig(path=str(database)),
        "skip_embeddings": True, "table_workers": 1, "neo4j": _neo4j(),
        "concept_spec": str(dictionary.spec), "concept_staging": str(dictionary.staging),
    })
    with fake_model_environment():
        report = run_build(config)
    if report["tables_failed"]:
        split.works.error = f"tables failed: {report['failed_tables']}"
        return
    _read(split.works)
    with split.works.driver.session() as session:
        for r in session.run(_COLUMNS, schema=ANNEX_SCHEMA):
            split.annex_refs[(r["t"], r["c"])] = r["ref"]
        for r in session.run(_LINKS, schema=ANNEX_SCHEMA):
            split.annex_links.setdefault((r["t"], r["c"]), []).append(dict(r["props"]))


@pytest.fixture(scope="module")
def split(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Split]:
    neo4j = _neo4j()
    driver = GraphDatabase.driver(neo4j.uri, auth=(neo4j.user, neo4j.password.get_secret_value()))
    built = Split(Graph(driver))
    try:
        _build_split(built, tmp_path_factory.mktemp("coded_split"))
    except Exception as exc:  # a build crash fails every test with its cause, never an error
        built.works.error = f"{type(exc).__name__}: {exc}"
    yield built
    driver.close()


def _unplaced(graph: Graph, prop: str) -> list[Evidence]:
    return [e for found in graph.built().evidence.values() for e in found
            if e.kind == "stated_domain" and e.payload.get("property") == prop
            and e.payload.get("linked") is False and "normalized name" in e.payload["reason"]]


def test_a_domain_placed_in_one_schema_is_not_unplaced_in_another(split: Split) -> None:
    assert _unplaced(split.works, SPLIT_UNPLACED_PROPERTY), "the unplaced recorder never ran"
    links = split.annex_links.get(("vat", "bevel_kind"), [])
    assert [k.get("verification") for k in links] == ["verified"]
    assert _unplaced(split.works, SPLIT_PLACED_PROPERTY) == []


def test_an_entity_in_one_schema_limits_placement_in_every_schema(split: Split) -> None:
    column = ("kiln", "seal_kind")
    _one_verified(split.works.built(), column, SEAL_KINDS)
    assert "stated_domain" in split.works.kinds(column)
    annex = split.annex_refs[("vat", "seal_kind")]
    assert split.annex_links.get(("vat", "seal_kind"), []) == []
    assert not any(e.kind == "stated_domain" for e in split.works.evidence.get(annex, []))
