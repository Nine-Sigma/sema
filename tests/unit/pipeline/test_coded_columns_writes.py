"""What the coded-column step writes, read back from a stand-in for the graph.

The step runs over the neutral fixture in DuckDB, through the real connector; only the graph
is a stand-in. It answers the step's reads (the run's columns, the declared keys, a value
set's current members) and keeps its writes the way the loader's Cypher merges them: a
`Term` per vocabulary and code whose label a write replaces only with a higher confidence and a
label of its own, `MEMBER_OF` per term and value set, `HAS_VALUE_SET` per column, and
`LABELED_BY` per pair.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from sema.connectors.duckdb import DuckDBConnector
from sema.graph.term_identity_utils import term_namespace
from sema.models.assertions import Assertion
from sema.models.config import BuildConfig, DuckDBConfig
from sema.models.extraction import ExtractedColumn, ExtractedForeignKey
from sema.pipeline.coded_columns import run_coded_columns
from sema.pipeline.normalizer import AssertionNormalizer
from tests.support.connector_fixture import FixtureTable, write_fixture_database
from tests.support.connector_fixture_codes import (
    ANNEX_SCHEMA,
    CODED_TABLES,
    CODES_CATALOG,
    CODES_SCHEMA,
    CROSS_SCHEMA_TABLES,
    GRIT,
    LATCH,
    NOTCH,
    NULL_LABEL_TABLES,
    TONE_NAMES,
)

pytestmark = pytest.mark.unit

RUN_ID = "coded-writes"
SchemaTables = dict[str, tuple[FixtureTable, ...]]
Column = tuple[str, str, str]


@dataclass
class Graph:
    """The graph stand-in: answers reads from `columns` and `joins`, keeps every write."""

    columns: list[ExtractedColumn]
    joins: dict[str, list[Assertion]]
    terms: dict[tuple[str, str], tuple[str | None, float]] = field(default_factory=dict)
    members: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    edges: dict[str, dict[str, Any]] = field(default_factory=dict)
    labeled_by: dict[str, list[tuple[str, dict[str, Any]]]] = field(default_factory=dict)
    assertions: list[Assertion] = field(default_factory=list)

    def _column(self, value_set_ref: str | None) -> str:
        """A value set is keyed by its column's dotted name; the stand-in keys it by the ref."""
        dotted = {f"{c.catalog}.{c.schema}.{c.table_name}.{c.name}": c.ref for c in self.columns}
        return dotted[value_set_ref or ""] or ""

    def _run_read(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "column_ref" in params:
            held = self.members.get(self._column(params["column_ref"]), ())
            return [{"code": code, "vocabulary_name": vocabulary}
                    for vocabulary, code in sorted(held)]
        if "predicates" in params:
            return [{**a.model_dump(), "payload": json.dumps(a.payload)}
                    for a in self.joins.get(params["schema"], [])]
        if "schema_name" in params:
            return [{"name": c.name, "table_name": c.table_name, "catalog": c.catalog,
                     "schema_name": c.schema, "data_type": c.data_type, "nullable": c.nullable,
                     "comment": c.comment, "ref": c.ref}
                    for c in self.columns if c.schema == params["schema_name"]]
        raise AssertionError(f"unexpected read: {cypher}")

    def _run(self, cypher: str, **params: Any) -> None:
        if "label_ref" in params:
            self.labeled_by.setdefault(params["column_ref"], []).append(
                (params["label_ref"], dict(params["edge"])))
            return
        if "rows" in params and "Term" in cypher:
            assert "r.label IS NOT NULL" in cypher, "the stand-in's label rule left the term write"
            for r in params["rows"]:
                self.term(r["vocabulary_name"], r["code"], r["label"], r["confidence"])
            return
        raise AssertionError(f"unexpected write: {cypher}")

    def term(self, vocabulary: str, code: str, label: str | None, confidence: float) -> None:
        held = self.terms.get((vocabulary, code))
        if held is None or (confidence > held[1] and (label is not None or held[0] is None)):
            self.terms[(vocabulary, code)] = (label, confidence)

    def store_assertion(self, assertion: Assertion, source_schema: str | None = None) -> None:
        self.assertions.append(assertion)

    def upsert_value_set(self, *_: Any, column_ref: str | None = None,
                         edge: dict[str, Any] | None = None, **__: Any) -> None:
        self.edges.setdefault(self._column(column_ref), {}).update(edge or {})

    def add_term_to_value_set(self, term_code: str, _name: str, *, vocabulary_name: str | None,
                              value_set_ref: str | None, **__: Any) -> None:
        member = (term_namespace(vocabulary_name), term_code)
        self.members.setdefault(self._column(value_set_ref), set()).add(member)

    def labels(self, ref: str) -> list[tuple[str, str | None]]:
        """Each member of the column's value set: its code and its term's label."""
        return sorted((code, self.terms.get((vocabulary, code), (None, 0.0))[0])
                      for vocabulary, code in self.members.get(ref, ()))

    def evidence(self, ref: str) -> list[dict[str, Any]]:
        return [{**a.payload, "object_ref": a.object_ref}
                for a in self.assertions if a.subject_ref == ref]


@dataclass
class Build:
    graph: Graph
    refs: dict[Column, str]


Fails = Callable[[str], bool]


def _never(_: str) -> bool:
    return False


def _connector(database: Path, fails: Fails) -> DuckDBConnector:
    connector = DuckDBConnector(DuckDBConfig(path=str(database)))
    reads = connector.execute_query

    def execute_query(sql: str) -> list[tuple[Any, ...]]:
        if fails(sql):
            raise RuntimeError("the warehouse refused the read")
        return reads(sql)

    connector.execute_query = execute_query  # type: ignore[method-assign]
    return connector


def build(
    directory: Path, tables: SchemaTables, read: list[str],
    stated_keys: Iterable[ExtractedForeignKey] = (),
    seeded: Callable[[Graph, dict[Column, str]], None] | None = None, fails: Fails = _never,
) -> Build:
    """Run the step over `tables`, reading the schemas in `read`, with keys a test states."""
    database = write_fixture_database(directory, tables, CODES_CATALOG)
    connector = _connector(database, fails)
    _, platform, workspace = connector.get_datasource_ref()
    normalizer = AssertionNormalizer(workspace, RUN_ID, platform)
    every = {schema: [t.name for t in tables[schema]] for schema in tables}
    columns = [
        replace(c, ref=normalizer.normalize_column(c)[0].subject_ref)
        for schema, names in every.items() for name in names
        for c in connector.extract_columns_dto(CODES_CATALOG, schema, name)
    ]
    keys = [k for schema in read for name in every[schema]
            for k in connector.extract_foreign_keys_dto(CODES_CATALOG, schema, name)]
    joins: dict[str, list[Assertion]] = {}
    for key in (*keys, *stated_keys):
        joins.setdefault(key.referencing_schema, []).extend(normalizer.normalize_foreign_key(key))
    graph = Graph([c for c in columns if c.schema in read], joins)
    refs = {(c.schema, c.table_name, c.name): c.ref or "" for c in columns}
    if seeded is not None:
        seeded(graph, refs)
    config = BuildConfig(source="duckdb", catalog=CODES_CATALOG, schemas=read,
                         duckdb=DuckDBConfig(path=str(database)))
    run_coded_columns(graph, connector, config, read, RUN_ID)  # type: ignore[arg-type]
    connector.close()
    return Build(graph, refs)


def works(table: str, column: str) -> Column:
    return (CODES_SCHEMA, table, column)


# Defect 1: an unverified binding writes no members; a value set holds each code once.

L2_VOCABULARY = "fitting_tone_code_values"
L2_CONFIDENCE = 0.8


def _decoded_by_l2(graph: Graph, refs: dict[Column, str]) -> None:
    """The model's decoded values on the column's value set, before the step runs."""
    ref = refs[works("fitting", "tone_code")]
    for code in TONE_NAMES:
        graph.term(L2_VOCABULARY, code, f"decoded {code}", L2_CONFIDENCE)
        graph.members.setdefault(ref, set()).add((L2_VOCABULARY, code))


@pytest.fixture(scope="module")
def coded(tmp_path_factory: pytest.TempPathFactory) -> Build:
    return build(tmp_path_factory.mktemp("coded_writes"), {CODES_SCHEMA: CODED_TABLES},
                 [CODES_SCHEMA], seeded=_decoded_by_l2)


def test_an_unverified_binding_writes_its_numbers_and_no_member(coded: Build) -> None:
    shift = coded.refs[works("panel", "shift")]
    assert coded.graph.edges[shift]["verification"] == "unverified"
    assert coded.graph.edges[shift]["listed_value_share"] == 1.0
    assert coded.graph.members.get(shift, set()) == set()
    finish = coded.refs[works("panel", "finish_code")]
    assert coded.graph.edges[finish]["verification"] == "verified"
    assert coded.graph.members[finish], "a verified binding writes its members"


def test_a_verified_binding_beside_decoded_values_holds_each_code_once(coded: Build) -> None:
    labels = coded.graph.labels(coded.refs[works("fitting", "tone_code")])
    assert Counter(code for code, _ in labels) == Counter(set(TONE_NAMES))
    assert dict(labels) == TONE_NAMES


DECODED_CONFIDENCE = 0.75


def _grit_decoded(graph: Graph, refs: dict[Column, str]) -> None:
    """Decoded labels on a column whose code relation has several text columns, so no label."""
    ref = refs[works("panel", "grit_code")]
    for code in GRIT:
        graph.term(L2_VOCABULARY, code, f"decoded {code}", DECODED_CONFIDENCE)
        graph.members.setdefault(ref, set()).add((L2_VOCABULARY, code))


def test_a_verified_binding_with_no_label_keeps_the_decoded_labels(tmp_path: Path) -> None:
    built = build(tmp_path, {CODES_SCHEMA: CODED_TABLES}, [CODES_SCHEMA], seeded=_grit_decoded)
    grit = built.refs[works("panel", "grit_code")]
    assert built.graph.edges[grit]["verification"] == "verified"
    assert built.graph.labeled_by.get(grit, []) == []
    assert dict(built.graph.labels(grit)) == {code: f"decoded {code}" for code in GRIT}


# Defect 2: a declared key resolves by its schema-qualified reference.

def _key_across_schemas() -> ExtractedForeignKey:
    return ExtractedForeignKey(
        constraint_name="bin_latch", referencing_catalog=CODES_CATALOG,
        referencing_schema=CODES_SCHEMA, referencing_table="bin", referenced_catalog=CODES_CATALOG,
        referenced_schema=ANNEX_SCHEMA, referenced_table="latch",
        column_pairs=(("latch_ref", "latch_code"),),
    )


@pytest.mark.parametrize("read", [[CODES_SCHEMA, ANNEX_SCHEMA], [CODES_SCHEMA]])
def test_a_declared_key_names_the_relation_of_its_own_schema(
    tmp_path: Path, read: list[str],
) -> None:
    built = build(tmp_path, CROSS_SCHEMA_TABLES, read, [_key_across_schemas()])
    column = built.refs[works("bin", "latch_ref")]
    referenced = built.refs[(ANNEX_SCHEMA, "latch", "latch_code")]
    declared = [e for e in built.graph.evidence(column) if e["evidence"] == "declared_key"]
    assert [e["object_ref"] for e in declared] == [referenced]
    if ANNEX_SCHEMA in read:
        assert built.graph.edges[column]["verification"] == "verified"
        assert dict(built.graph.labels(column)) == LATCH
        ((label, edge),) = built.graph.labeled_by.get(column, [])
        assert label == built.refs[(ANNEX_SCHEMA, "latch", "latch_title")]
        assert edge["verification"] == "verified" and edge["via"]
        return
    assert declared[0]["linked"] is False
    assert "not read" in declared[0]["reason"]
    assert column not in built.graph.edges


# Defect 3: a label column is determined by its non-NULL texts, and a failed read is no label.

@pytest.fixture(scope="module")
def null_labels(tmp_path_factory: pytest.TempPathFactory) -> Build:
    return build(tmp_path_factory.mktemp("null_labels"), {CODES_SCHEMA: NULL_LABEL_TABLES},
                 [CODES_SCHEMA])


def test_a_mostly_null_text_column_is_no_label(null_labels: Build) -> None:
    column = null_labels.refs[works("ledge", "notch_code")]
    assert null_labels.graph.edges[column]["verification"] == "verified"
    ((label, edge),) = null_labels.graph.labeled_by.get(column, [])
    assert label == null_labels.refs[works("ledge", "notch_title")]
    assert edge == {"verification": "verified", "violation_rate": 0.0, "via": []}
    assert dict(null_labels.graph.labels(column)) == {c: t for c, (_, t) in NOTCH.items()}


def _pairs_of_finish(sql: str) -> bool:
    return "`finish_code`" in sql and "`finish_label`" in sql


@pytest.mark.parametrize("fails", [_never, _pairs_of_finish])
def test_a_failed_label_read_writes_no_label(tmp_path: Path, fails: Fails) -> None:
    built = build(tmp_path, {CODES_SCHEMA: CODED_TABLES}, [CODES_SCHEMA], fails=fails)
    column = built.refs[works("panel", "finish_code")]
    assert built.graph.edges[column]["verification"] == "verified"
    labeled = [label for label, edge in built.graph.labeled_by.get(column, [])
               if edge["verification"] == "verified"]
    if fails is _never:
        assert labeled == [built.refs[works("finish", "finish_label")]]
    else:
        assert labeled == []
