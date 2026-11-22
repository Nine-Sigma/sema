"""Running the identity fixture: a model stand-in, the build config, and a stand-in graph
for one build step.

`IdentityFakeClient` names the L2 entities the fixture states and answers a proposal call
with the fixture's proposal for the relation the call names. `StepGraph` stands in for the
graph a build step reads and writes: it answers the run's columns, its declared joins and a
value set's members, and keeps every write, so a unit test runs a step over the fixture in
DuckDB, through the real connector, and reads back what the step wrote.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import yaml
from pydantic import SecretStr

from sema.connectors.base import StatementTimeout
from sema.connectors.duckdb import DuckDBConnector
from sema.graph.term_identity_utils import term_namespace
from sema.models.assertions import Assertion
from sema.models.config import BuildConfig, DuckDBConfig, Neo4jConfig
from sema.models.extraction import ExtractedColumn
from sema.models.physical_key import CanonicalRef
from sema.models.stages import StageAResult
from sema.pipeline.normalizer import AssertionNormalizer
from tests.support.connector_fixture import FakeLLMClient
from tests.support.connector_fixture_codes import DictionaryFiles
from tests.support.connector_fixture_identity import (
    IDENTITY_CATALOG,
    IDENTITY_SCHEMA,
    L2_ENTITIES,
    PROPOSALS,
    TABLES,
    conflict_shares,
    distinct_count,
    null_row_count,
    write_identity_database,
    write_identity_dictionary,
)

STEP_RUN_ID = "fixident-step-run"
Column = tuple[str, str]
TimesOut = Callable[[str], bool]


def identity_build_config(
    database: Path, dictionary: DictionaryFiles, neo4j: Neo4jConfig | None = None,
) -> BuildConfig:
    """The build over the fixture through the DuckDB connector, with the dictionary."""
    config = BuildConfig(
        source="duckdb", catalog=IDENTITY_CATALOG, schemas=[IDENTITY_SCHEMA],
        duckdb=DuckDBConfig(path=str(database)), skip_embeddings=True, table_workers=1,
        concept_spec=str(dictionary.spec), concept_staging=str(dictionary.staging),
    )
    return config if neo4j is None else config.model_copy(update={"neo4j": neo4j})


def relation_named(text: str, table_ref: str = "") -> str | None:
    """The fixture relation a model call is about: its table ref's last part, else the
    relation the prompt names first as a whole word."""
    last = table_ref.rstrip("/").rsplit("/", 1)[-1]
    if last in TABLES:
        return last
    found = [(m.start(), name) for name in TABLES
             if (m := re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text))]
    return min(found)[1] if found else None


def proposal_for(schema: Any, prompt: str, table_ref: str = "") -> Any:
    """The fixture's proposal for the relation a call names, as the schema the caller asked."""
    proposal = PROPOSALS.get(relation_named(prompt, table_ref) or "")
    if proposal is None:
        return None
    return schema(entity=proposal.entity, entity_key=list(proposal.entity_key),
                  latest_by=proposal.latest_by,
                  row_key=list(proposal.row_key) if proposal.row_key else None)


def _is_proposal_schema(schema: Any) -> bool:
    return getattr(schema, "__name__", "") == "IdentityProposal"


class IdentityFakeClient(FakeLLMClient):
    """The connector fixture's fake, with the L2 entity names above and the proposals."""

    def invoke(  # type: ignore[override]
        self, prompt: str, schema: Any, *, table_ref: str = "", stage_name: str = "",
        simplified_prompt: str | None = None,
    ) -> Any:
        table = table_ref.rstrip("/").rsplit("/", 1)[-1]
        if schema is StageAResult and table in L2_ENTITIES:
            return StageAResult(primary_entity=L2_ENTITIES[table],
                                grain_hypothesis=f"one row per {table}", confidence=0.9)
        if _is_proposal_schema(schema):
            return proposal_for(schema, prompt, table_ref)
        return super().invoke(prompt, schema, table_ref=table_ref, stage_name=stage_name,
                              simplified_prompt=simplified_prompt)


@contextmanager
def identity_model_environment() -> Iterator[None]:
    """Route every model call the build makes to `IdentityFakeClient`."""
    with patch("sema.pipeline.build.LLMClientFactory.create",
               lambda _self: IdentityFakeClient(llm=MagicMock())):
        yield


def _never(_: str) -> bool:
    return False


def fixture_connector(database: Path, times_out: TimesOut = _never) -> DuckDBConnector:
    """The DuckDB connector over the fixture; a statement `times_out` names raises the
    connector's timeout from either read."""
    connector = DuckDBConnector(DuckDBConfig(path=str(database)))
    plain, timed = connector.execute_query, connector.execute_query_with_columns

    def execute_query(sql: str) -> list[tuple[Any, ...]]:
        if times_out(sql):
            raise StatementTimeout(sql)
        return plain(sql)

    def execute_query_with_columns(sql: str, **bounds: Any) -> Any:
        if times_out(sql):
            raise StatementTimeout(sql)
        return timed(sql, **bounds)

    connector.execute_query = execute_query  # type: ignore[method-assign]
    connector.execute_query_with_columns = execute_query_with_columns  # type: ignore[method-assign]
    return connector


@dataclass
class StepGraph:
    """The graph stand-in: answers reads from `columns` and `joins`, keeps every write.

    A read it does not know returns no rows; a write it does not know is kept in `writes`.
    """

    columns: list[ExtractedColumn]
    joins: list[Assertion]
    terms: dict[tuple[str, str], tuple[str | None, float]] = field(default_factory=dict)
    members: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    edges: dict[str, dict[str, Any]] = field(default_factory=dict)
    labeled_by: dict[str, list[tuple[str, dict[str, Any]]]] = field(default_factory=dict)
    assertions: list[Assertion] = field(default_factory=list)
    writes: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    def _column(self, value_set_ref: str | None) -> str:
        dotted = {f"{c.catalog}.{c.schema}.{c.table_name}.{c.name}": c.ref for c in self.columns}
        return dotted.get(value_set_ref or "") or value_set_ref or ""

    def _run_read(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "column_ref" in params:
            held = self.members.get(self._column(params["column_ref"]), ())
            return [{"code": code, "vocabulary_name": vocabulary}
                    for vocabulary, code in sorted(held)]
        if "predicates" in params:
            return [{**a.model_dump(), "payload": json.dumps(a.payload)} for a in self.joins]
        if "schema_name" in params:
            return [{"name": c.name, "table_name": c.table_name, "catalog": c.catalog,
                     "schema_name": c.schema, "data_type": c.data_type, "nullable": c.nullable,
                     "comment": c.comment, "ref": c.ref}
                    for c in self.columns if c.schema == params["schema_name"]]
        return []

    def _run(self, cypher: str, **params: Any) -> None:
        if "label_ref" in params:
            self.labeled_by.setdefault(params["column_ref"], []).append(
                (params["label_ref"], dict(params["edge"])))
        elif "rows" in params and "Term" in cypher and "r.label IS NOT NULL" in cypher:
            for r in params["rows"]:
                self.term(r["vocabulary_name"], r["code"], r["label"], r["confidence"])
        else:
            self.writes.append(("_run", (cypher,), params))

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

    def __getattr__(self, name: str) -> Callable[..., None]:
        if name.startswith("_"):
            raise AttributeError(name)

        def record(*args: Any, **kwargs: Any) -> None:
            self.writes.append((name, args, kwargs))

        return record

    def labels(self, ref: str) -> dict[str, str | None]:
        """Each member of the column's value set: its code and its term's label."""
        return {code: self.terms.get((vocabulary, code), (None, 0.0))[0]
                for vocabulary, code in self.members.get(ref, ())}

    def asserted(self, predicate: str) -> list[Assertion]:
        return [a for a in self.assertions if a.predicate.value == predicate]


def relation_of(ref: str) -> str:
    """The relation a table or column reference names."""
    return CanonicalRef.parse(ref).table


@dataclass
class StepRun:
    graph: StepGraph
    connector: DuckDBConnector
    config: BuildConfig
    refs: dict[Column, str]
    dictionary: DictionaryFiles
    error: str | None = None

    def built(self) -> StepGraph:
        """The graph the step wrote; a step that crashed fails the test with its cause."""
        assert self.error is None, self.error
        return self.graph


def run_step(prepared: StepRun, step: Callable[[StepRun], Any]) -> StepRun:
    """Run one step on a prepared fixture; a crash is kept, so each test fails, never errors."""
    try:
        step(prepared)
    except Exception as exc:  # each test reports the step's crash as its own failure
        prepared.error = f"{type(exc).__name__}: {exc}"
    finally:
        prepared.connector.close()
    return prepared


def prepare_step(directory: Path, times_out: TimesOut = _never) -> StepRun:
    """The fixture in DuckDB, its columns and declared joins in a stand-in graph, and the
    build config with the dictionary, ready for one build step; only the step's own reads
    meet `times_out`."""
    database = write_identity_database(directory)
    dictionary = write_identity_dictionary(directory)
    reader = fixture_connector(database)
    try:
        _, platform, workspace = reader.get_datasource_ref()
        normalizer = AssertionNormalizer(workspace, STEP_RUN_ID, platform)
        columns = [
            replace(c, ref=normalizer.normalize_column(c)[0].subject_ref)
            for name in TABLES
            for c in reader.extract_columns_dto(IDENTITY_CATALOG, IDENTITY_SCHEMA, name)
        ]
        joins = [a for name in TABLES
                 for key in reader.extract_foreign_keys_dto(IDENTITY_CATALOG, IDENTITY_SCHEMA, name)
                 for a in normalizer.normalize_foreign_key(key)]
    finally:
        reader.close()
    refs = {(c.table_name, c.name): c.ref or "" for c in columns}
    return StepRun(StepGraph(columns, joins), fixture_connector(database, times_out),
                   identity_build_config(database, dictionary), refs, dictionary)


# The facts the fixture's rows and dictionary hold, by case: each relation's key, the unique
# set with a NULL part, the entities coarser than their row, the shared entity, the labels.
KEYED: dict[str, tuple[str, ...]] = {
    "dock_kind": ("dock_code",), "rig_kind": ("rig_code",), "berth": ("berth_no",),
    "hull_rev": ("hull_no", "rev_no"), "hull_survey": ("survey_no",),
    "hull_cargo": ("cargo_no",), "hull_quote": ("quote_no",), "tide_log": ("log_no",),
    "buoy_check": ("check_no",), "crane_rev": ("crane_rev_id",), "pump": ("unit_no",),
    "valve": ("unit_no",), "slip_a": ("slip_id",), "slip_b": ("slip_ref",),
    "slip_use": ("use_no",), "berth_call": ("call_no",), "deck_plan": ("plan_no",),
}
NULL_PART_KEYS: dict[str, tuple[str, ...]] = {"mooring": ("pier_code", "slot_no")}
COARSER: dict[str, tuple[str, ...]] = {
    "hull_rev": ("hull_no",), "crane_rev": ("crane_no",), "buoy_check": ("buoy_no",),
    "berth_call": ("vessel_no",),
}
SHARED: dict[tuple[str, tuple[str, ...]], tuple[str, ...]] = {
    ("hull_rev", ("hull_no",)): ("hull_cargo", "hull_survey"),
}
LABELED: dict[Column, Column | None] = {
    ("hull_rev", "rig_type"): ("rig_kind", "rig_title"), ("hull_cargo", "hatch_kind"): None,
}


def columns_key(columns: tuple[str, ...]) -> str:
    return "+".join(sorted(columns))


def ordered_by(table: str) -> str | None:
    """The proposal's ordering column when it picks exactly one row of each entity."""
    by, key = PROPOSALS[table].latest_by, COARSER[table]
    picks_one = by is not None and distinct_count(table, (*key, by)) == len(TABLES[table].rows)
    return by if picks_one else None


def attribute_rules(table: str) -> dict[str, str]:
    """Each non-key column's rule: single with no conflict, else the ordering when it holds."""
    by, rules = ordered_by(table), {}
    for column, share in conflict_shares(table, COARSER[table]).items():
        if share == 0:
            rules[column] = "single"
        elif by is not None:
            rules[column] = f"latest:{by}"
    return rules


def identity_claims() -> dict[str, dict[tuple[str, ...], Any]]:
    """The fixture's facts per slice C kind, in the contract's F1 forms. The kind ids and value
    words live here, not in a constant, so the literal guard does not read them."""
    nulls = [*KEYED.items(), *NULL_PART_KEYS.items(), *COARSER.items()]
    return {
        "candidate_key": {(t, columns_key(k)): "unique" for t, k in KEYED.items()},
        "entity_coarser_than_row": {(t,): columns_key(k) for t, k in COARSER.items()},
        "entity_shared_across_relations": {
            (owner, columns_key(key)): sorted(others) for (owner, key), others in SHARED.items()},
        "null_key_policy": {(t, columns_key(k)): null_row_count(t, k) for t, k in nulls},
        "attribute_resolution_rule": {
            (t, column): rule for t in COARSER for column, rule in attribute_rules(t).items()},
        "label_source": {key: list(label) if label else "dictionary_only"
                         for key, label in LABELED.items()},
    }


def neo4j_config() -> Neo4jConfig:
    return Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=SecretStr(os.getenv("NEO4J_PASSWORD", "graphrag")),
    )


@dataclass
class IdentityBuild:
    """The fixture built into the graph: its build config file, its database, and why the
    build failed, if it did."""

    config_path: Path
    database: Path
    error: str | None = None

    def built(self) -> IdentityBuild:
        assert self.error is None, self.error
        return self


def build_identity_graph(driver: Any, directory: Path) -> IdentityBuild:
    """Clear the graph, then build the fixture with its dictionary and the mocked model."""
    from sema.pipeline.orchestrate import run_build

    database, dictionary = write_identity_database(directory), write_identity_dictionary(directory)
    neo4j = neo4j_config()
    config_path = directory / "identity_build.yaml"
    settings = identity_build_config(database, dictionary).model_dump(
        mode="json", include={"source", "catalog", "schemas", "duckdb", "skip_embeddings",
                              "table_workers", "concept_spec", "concept_staging"})
    settings["neo4j"] = {"uri": neo4j.uri, "user": neo4j.user,
                         "password": neo4j.password.get_secret_value()}
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    build = IdentityBuild(config_path, database)
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    try:
        with identity_model_environment():
            report = run_build(BuildConfig.from_file(str(config_path)))
    except Exception as exc:  # a build crash fails every test with its cause, never an error
        build.error = f"{type(exc).__name__}: {exc}"
        return build
    if report["tables_failed"]:
        build.error = f"tables failed: {report['failed_tables']}"
    return build
