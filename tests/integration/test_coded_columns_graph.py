"""Done-when 1: coded columns, value sets and labels in the graph the build writes.

Built through the DuckDB connector with the dictionary's concept spec, model mocked, no
embeddings. Expected numbers come from the fixture's rows; one test per case or decoy.
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from neo4j import Driver, GraphDatabase
from pydantic import SecretStr

from sema.consumers.nl2sql.prompting import build_sql_prompt
from sema.models.config import BuildConfig, DuckDBConfig, Neo4jConfig
from sema.pipeline.context import prune_to_sco
from sema.pipeline.orchestrate import run_build
from sema.pipeline.retrieval import RetrievalEngine
from tests.support.connector_fixture import fake_model_environment
from tests.support.connector_fixture_codes import (
    CODE_LISTS,
    CODES_CATALOG,
    CODES_SCHEMA,
    CREW,
    FINISH,
    GLAZE,
    GRIT,
    SHEEN,
    SHIFT_SLOT,
    TONE_NAMES,
    UNPLACED_PROPERTY,
    column_values,
    write_coded_database,
    write_dictionary,
)

pytestmark = pytest.mark.integration

Column = tuple[str, str]
DICTIONARY_FIELDS = {"concept_spec", "concept_staging"}


@dataclass(frozen=True)
class Link:
    value_set: str
    props: dict[str, Any]
    members: dict[str, tuple[str | None, int | None, bool | None]]


@dataclass(frozen=True)
class Evidence:
    predicate: str
    payload: dict[str, Any]
    object_ref: str | None

    @property
    def kind(self) -> Any:
        return self.payload.get("evidence")


@dataclass
class Graph:
    driver: Driver
    error: str | None = None
    refs: dict[Column, str] = field(default_factory=dict)
    links: dict[Column, list[Link]] = field(default_factory=dict)
    labeled_by: dict[Column, list[tuple[Column, dict[str, Any]]]] = field(default_factory=dict)
    evidence: dict[str, list[Evidence]] = field(default_factory=dict)
    join_paths: dict[str, set[str]] = field(default_factory=dict)

    def built(self) -> Graph:
        assert self.error is None, self.error
        return self

    def verified(self, column: Column) -> list[Link]:
        return [k for k in self.links.get(column, []) if k.props.get("verification") == "verified"]

    def evidence_of(self, column: Column) -> list[Evidence]:
        return self.evidence.get(self.refs[column], [])

    def kinds(self, column: Column) -> set[Any]:
        return {e.kind for e in self.evidence_of(column)}

    def containment(self, column: Column) -> list[Evidence]:
        return [e for e in self.evidence_of(column) if e.kind == "containment"]

    def objects(self, column: Column, kind: str) -> set[str | None]:
        return {e.object_ref for e in self.evidence_of(column) if e.kind == kind}

    def verified_labels(self, column: Column) -> list[tuple[Column, dict[str, Any]]]:
        return [(label, props) for label, props in self.labeled_by.get(column, [])
                if props.get("verification") == "verified"]


def _payload(raw: Any) -> dict[str, Any]:
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    return parsed if isinstance(parsed, dict) else {}


def _rows(driver: Driver, query: str) -> list[dict[str, Any]]:
    with driver.session() as session:
        return [record.data() for record in session.run(query, schema=CODES_SCHEMA)]


_LINKS = (
    "MATCH (c:Column {schema_name: $schema})-[h:HAS_VALUE_SET]->(vs:ValueSet) "
    "OPTIONAL MATCH (t:Term)-[m:MEMBER_OF]->(vs) "
    "RETURN c.table_name AS t, c.name AS c, elementId(vs) AS vs, properties(h) AS props, "
    "collect([t.code, t.label, m.rows, m.listed]) AS members"
)
_LABELED_BY = (
    "MATCH (c:Column {schema_name: $schema})-[l:LABELED_BY]->(lc:Column) "
    "RETURN c.table_name AS t, c.name AS c, lc.table_name AS lt, lc.name AS lc, "
    "properties(l) AS props"
)
_COLUMNS = (
    "MATCH (c:Column {schema_name: $schema}) "
    "RETURN c.table_name AS t, c.name AS c, c.ref AS ref"
)
_ASSERTIONS = (
    "MATCH (a:Assertion) RETURN a.subject_ref AS subject, a.predicate AS predicate, "
    "a.payload AS payload, a.object_ref AS object_ref"
)
_JOIN_PATHS = (
    "MATCH (jp:JoinPath)-[:USES]->(t:Table) "
    "RETURN jp.name AS name, collect(t.name) AS tables"
)


def _read(graph: Graph) -> None:
    driver = graph.driver
    graph.refs = {(r["t"], r["c"]): r["ref"] for r in _rows(driver, _COLUMNS)}
    for r in _rows(driver, _LINKS):
        members = {str(code): (label, rows, listed) for code, label, rows, listed in r["members"]
                   if code is not None}
        graph.links.setdefault((r["t"], r["c"]), []).append(Link(r["vs"], r["props"], members))
    for r in _rows(driver, _LABELED_BY):
        graph.labeled_by.setdefault((r["t"], r["c"]), []).append(((r["lt"], r["lc"]), r["props"]))
    for r in _rows(driver, _ASSERTIONS):
        evidence = Evidence(r["predicate"], _payload(r["payload"]), r["object_ref"])
        graph.evidence.setdefault(r["subject"], []).append(evidence)
    graph.join_paths = {r["name"]: set(r["tables"]) for r in _rows(driver, _JOIN_PATHS)}


def _build(graph: Graph, directory: Path) -> None:
    with graph.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    database = write_coded_database(directory)
    dictionary = write_dictionary(directory)
    config = BuildConfig(
        source="duckdb", catalog=CODES_CATALOG, schemas=[CODES_SCHEMA],
        duckdb=DuckDBConfig(path=str(database)), skip_embeddings=True, table_workers=1,
        neo4j=Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=SecretStr(os.getenv("NEO4J_PASSWORD", "graphrag")),
        ),
    )
    if DICTIONARY_FIELDS <= set(BuildConfig.model_fields):
        config = config.model_copy(update={
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
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "graphrag")),
    )
    built = Graph(driver)
    try:
        _build(built, tmp_path_factory.mktemp("coded"))
    except Exception as exc:  # a build crash fails every test with its cause, never an error
        built.error = f"{type(exc).__name__}: {exc}"
    yield built
    driver.close()


def _coverage(column: Column, listed: Collection[object]) -> dict[str, Any]:
    values = column_values(*column)
    present = [str(v) for v in values if v is not None]
    observed, codes = set(present), {str(c) for c in listed}
    return {
        "observed_distinct": len(observed),
        "null_rows": values.count(None),
        "listed_value_share": pytest.approx(len(observed & codes) / len(observed)),
        "listed_row_share": pytest.approx(sum(v in codes for v in present) / len(present)),
    }


def _members(column: Column, labels: dict[str, str | None]) -> dict[str, tuple[Any, ...]]:
    present = [str(v) for v in column_values(*column) if v is not None]
    codes = {str(c) for c in labels} | set(present)
    return {c: (labels.get(c), present.count(c), c in {str(k) for k in labels}) for c in codes}


def _one_verified(graph: Graph, column: Column, listed: Collection[object]) -> Link:
    links = graph.verified(column)
    assert len(links) == 1, f"{column}: verified value sets {links}"
    expected = _coverage(column, listed)
    assert {k: links[0].props.get(k) for k in expected} == expected
    return links[0]


def test_the_build_reads_the_dictionary() -> None:
    assert DICTIONARY_FIELDS <= set(BuildConfig.model_fields), "the build takes no concept spec"


def test_sibling_label_by_declared_key(graph: Graph) -> None:
    column = ("fitting", "tone_code")
    link = _one_verified(graph.built(), column, TONE_NAMES)
    assert link.members == _members(column, dict(TONE_NAMES))
    ((label_column, props),) = graph.verified_labels(column)
    assert label_column == ("fitting", "tone_name")
    assert props.get("violation_rate") == 0.0
    assert not props.get("via")
    assert {"declared_key", "containment"} <= graph.kinds(column)


def test_label_relation_by_join_counts_nulls(graph: Graph) -> None:
    column = ("panel", "finish_code")
    link = _one_verified(graph.built(), column, FINISH)
    assert link.props["null_rows"] == 250
    assert link.members == _members(column, dict(FINISH))
    assert link.members["POL"] == ("Polished", 0, True)
    ((label_column, props),) = graph.verified_labels(column)
    assert label_column == ("finish", "finish_label")
    assert props.get("violation_rate") == 0.0
    via = props.get("via") or []
    assert via and all({"panel", "finish"} <= graph.join_paths.get(name, set()) for name in via)
    assert {"declared_key", "containment"} <= graph.kinds(column)


def test_dictionary_only_label_by_stated_domain_counts_unlisted_values(graph: Graph) -> None:
    column = ("panel", "coat_kind")
    link = _one_verified(graph.built(), column, CODE_LISTS["coat_kinds"])
    assert link.members == _members(column, dict(CODE_LISTS["coat_kinds"]))
    assert link.members["CK-Z"] == (None, column_values(*column).count("CK-Z"), False)
    assert link.members["CK-F"] == ("Anodised", 0, True)
    assert graph.verified_labels(column) == []
    assert {"stated_domain", "containment"} <= graph.kinds(column)


def test_data_only_containment_stays_unverified_with_its_numbers(graph: Graph) -> None:
    column = ("panel", "shift")
    links = graph.built().links.get(column, [])
    assert links, f"{column} has no value set"
    assert all(k.props.get("verification") == "unverified" for k in links)
    expected = _coverage(column, SHIFT_SLOT)
    assert any({k: link.props.get(k) for k in expected} == expected for link in links)
    assert "containment" in graph.kinds(column)
    assert not {"declared_key", "stated_domain", "name_match"} & graph.kinds(column)


def test_a_name_match_the_data_agrees_with_verifies(graph: Graph) -> None:
    column = ("fitting", "glaze_code")
    _one_verified(graph.built(), column, GLAZE)
    assert {"name_match", "containment"} <= graph.kinds(column)
    assert not {"declared_key", "stated_domain"} & graph.kinds(column)
    assert graph.objects(column, "name_match") == {graph.refs[("glaze", "glaze_code")]}


def test_several_text_columns_link_with_the_label_unset(graph: Graph) -> None:
    column = ("panel", "grit_code")
    link = _one_verified(graph.built(), column, GRIT)
    assert link.members == _members(column, dict.fromkeys(GRIT))
    assert graph.verified_labels(column) == []


def test_two_determined_sibling_columns_leave_the_label_unset(graph: Graph) -> None:
    column = ("fitting", "sheen_code")
    link = _one_verified(graph.built(), column, SHEEN)
    assert link.members == _members(column, dict.fromkeys(SHEEN))
    assert graph.verified_labels(column) == []


@pytest.mark.parametrize(
    ("decoy", "linked"),
    [(("fitting", "grit_code"), ("panel", "grit_code")),
     (("panel", "finish_cd"), ("panel", "finish_code"))],
)
def test_a_name_alike_column_outside_the_value_set_does_not_link(
    graph: Graph, decoy: Column, linked: Column,
) -> None:
    assert graph.built().verified(linked), f"{linked} has no verified value set"
    assert graph.verified(decoy) == []


def test_a_name_match_the_data_refutes_does_not_link(graph: Graph) -> None:
    column = ("fitting", "grit_code")
    assert graph.built().objects(column, "name_match") == {graph.refs[("grit", "grit_code")]}
    assert graph.verified(column) == []


@pytest.mark.parametrize(
    ("column", "vocabulary", "share"),
    [(("panel", "trim"), "trim_kinds", 1 / 7), (("fitting", "coat_kind"), "coat_kinds", 6 / 10)],
)
def test_a_stated_domain_below_the_floor_stays_unverified_with_its_numbers(
    graph: Graph, column: Column, vocabulary: str, share: float,
) -> None:
    assert graph.built().verified(column) == []
    assert "stated_domain" in graph.kinds(column)
    shares = [e.payload.get("listed_value_share") for e in graph.containment(column)]
    assert pytest.approx(share) in shares, f"{column} against {vocabulary}: {shares}"


def test_coincidental_containment_stays_unverified(graph: Graph) -> None:
    column = ("panel", "crew_size")
    assert graph.built().verified(column) == []
    shares = [e.payload.get("listed_value_share") for e in graph.containment(column)]
    assert 1.0 in shares


def test_a_tie_stays_unverified_with_both_alternatives(graph: Graph) -> None:
    column = ("panel", "rack_no")
    assert graph.built().verified(column) == []
    alternatives = {graph.refs[("rack", "rack_no")], graph.refs[("rack_spare", "rack_no")]}
    full = {e.object_ref for e in graph.containment(column)
            if e.payload.get("listed_value_share") == 1.0}
    assert alternatives <= full
    assert alternatives <= graph.objects(column, "name_match")


def test_a_column_and_a_code_relation_at_the_bound_link(graph: Graph) -> None:
    _one_verified(graph.built(), ("panel", "crew_ref"), CREW)
    assert {"name_match", "containment"} <= graph.kinds(("panel", "crew_ref"))


def test_a_column_past_the_cardinality_bound_is_not_nominated(graph: Graph) -> None:
    column = ("panel", "assigned_crew")
    assert graph.built().containment(("panel", "crew_size")), "the nominator never ran"
    assert graph.verified(("panel", "crew_ref")), "the relation at the bound never linked"
    assert graph.links.get(column, []) == []
    assert graph.containment(column) == []


def test_a_declared_key_to_a_relation_past_the_bound_is_no_value_set(graph: Graph) -> None:
    assert graph.built().verified(("panel", "finish_code")), "no declared key linked"
    assert graph.links.get(("panel", "pallet_no"), []) == []


@pytest.mark.parametrize(
    ("column", "vocabulary"),
    [(("panel", "seal_kind"), "seal_kinds"), (("fitting", "seal_kind"), "seal_kinds"),
     (("panel", "bevel_kind"), "bevel_kinds")],
)
def test_a_stated_domain_lands_on_each_normalized_name_match(
    graph: Graph, column: Column, vocabulary: str,
) -> None:
    link = _one_verified(graph.built(), column, CODE_LISTS[vocabulary])
    assert link.members == _members(column, dict(CODE_LISTS[vocabulary]))
    assert {"stated_domain", "containment"} <= graph.kinds(column)


def test_an_entity_naming_a_table_limits_placement_to_it(graph: Graph) -> None:
    assert "stated_domain" in graph.built().kinds(("panel", "trim"))
    assert "stated_domain" not in graph.kinds(("fitting", "trim"))
    assert graph.verified(("fitting", "trim")) == []


def test_an_unmatched_stated_domain_is_listed_as_unplaced(graph: Graph) -> None:
    unplaced = [
        e for found in graph.built().evidence.values() for e in found
        if e.kind == "stated_domain" and e.payload.get("property") == UNPLACED_PROPERTY
    ]
    assert unplaced and all(e.payload.get("reason") for e in unplaced), unplaced
    edge_codes = set(CODE_LISTS["edge_kinds"])
    assert not any(edge_codes & set(k.members) for links in graph.links.values() for k in links)


LABEL_QUESTIONS = {
    "sibling label column": ("How many fittings are Cobalt?", "tone_code", "T-BL", "Cobalt"),
    "label relation by join": ("Count the Brushed panels", "finish_code", "BRU", "Brushed"),
    "dictionary only": ("Which panels carry a Lacquer coat?", "coat_kind", "CK-C", "Lacquer"),
}


@pytest.mark.parametrize("location", list(LABEL_QUESTIONS))
def test_a_question_naming_a_label_reaches_the_column_and_the_code(
    graph: Graph, location: str,
) -> None:
    question, column, code, label = LABEL_QUESTIONS[location]
    sco = prune_to_sco(RetrievalEngine(driver=graph.built().driver).retrieve(question))
    prompt = build_sql_prompt(sco, question)
    lines = [line for line in prompt.splitlines() if column in line and code in line]
    assert any(label in line for line in lines), f"{location}: {lines}\n{prompt}"
