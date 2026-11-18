"""The core query path on a graph whose verified value set has members with no label.

The coded fixture is built through the DuckDB connector with the dictionary's concept spec,
model mocked, no embeddings. Its code relation with several text columns links its value
set and leaves every member's label unset. The mocked model types each coded column
`categorical`, as a real model reads one; the stock fake types every column `attribute`,
and entity expansion lists the values of categorical columns only. A question naming the
fact relation then reaches that value set through core retrieval, and the context renders
each member as its code with `no label`; the decoy column's members keep their labels.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from neo4j import Driver, GraphDatabase
from pydantic import SecretStr

from sema.consumers.nl2sql.prompting import build_sql_prompt
from sema.models.config import BuildConfig, DuckDBConfig, Neo4jConfig
from sema.models.stages import StageBBatchResult
from sema.pipeline.context import prune_to_sco
from sema.pipeline.orchestrate import run_build
from sema.pipeline.retrieval import RetrievalEngine
from tests.support.connector_fixture import FakeLLMClient
from tests.support.connector_fixture_codes import (
    CODES_CATALOG,
    CODES_SCHEMA,
    FINISH,
    GRIT,
    write_coded_database,
    write_dictionary,
)

pytestmark = pytest.mark.integration

QUESTION = "How many panel records are there?"
UNLABELLED = ("panel", "grit_code")
LABELLED = ("panel", "finish_code")
CODED_SUFFIX = "_code"
_MEMBERS = (
    "MATCH (c:Column {schema_name: $schema, table_name: $table, name: $column})"
    "-[h:HAS_VALUE_SET {verification: 'verified'}]->(:ValueSet)<-[:MEMBER_OF]-(t:Term) "
    "RETURN t.code AS code, t.label AS label"
)


class CodedColumnsAreCategorical(FakeLLMClient):
    def invoke(  # type: ignore[override]
        self, prompt: str, schema: Any, **kwargs: Any,
    ) -> Any:
        result = super().invoke(prompt, schema, **kwargs)
        if schema is not StageBBatchResult:
            return result
        return result.model_copy(update={"columns": [
            c.model_copy(update={"semantic_type": "categorical"})
            if c.column.endswith(CODED_SUFFIX) else c
            for c in result.columns
        ]})


def _neo4j() -> Neo4jConfig:
    return Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=SecretStr(os.getenv("NEO4J_PASSWORD", "graphrag")),
    )


def _build(driver: Driver, directory: Path) -> None:
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    dictionary = write_dictionary(directory)
    config = BuildConfig(
        source="duckdb", catalog=CODES_CATALOG, schemas=[CODES_SCHEMA],
        duckdb=DuckDBConfig(path=str(write_coded_database(directory))),
        skip_embeddings=True, table_workers=1, neo4j=_neo4j(),
        concept_spec=str(dictionary.spec), concept_staging=str(dictionary.staging),
    )
    model = "sema.pipeline.build.LLMClientFactory.create"
    with patch(model, lambda _self: CodedColumnsAreCategorical(llm=MagicMock())):
        report = run_build(config)
    assert not report["tables_failed"], report["failed_tables"]


@pytest.fixture(scope="module")
def driver(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Driver]:
    config = _neo4j()
    opened = GraphDatabase.driver(config.uri, auth=(config.user, config.password.get_secret_value()))
    _build(opened, tmp_path_factory.mktemp("coded_context"))
    yield opened
    opened.close()


def _members(driver: Driver, column: tuple[str, str]) -> dict[str, Any]:
    table, name = column
    with driver.session() as session:
        rows = session.run(_MEMBERS, schema=CODES_SCHEMA, table=table, column=name)
        return {str(r["code"]): r["label"] for r in rows}


def _governed_line(text: str, column: tuple[str, str]) -> str:
    table, name = column
    (line,) = [x for x in text.splitlines() if x.strip().startswith(f"{table}.{name}:")]
    return line


def test_the_verified_value_set_has_members_with_no_label(driver: Driver) -> None:
    members = _members(driver, UNLABELLED)
    assert set(GRIT) <= set(members), members
    assert all(members[code] is None for code in GRIT), members


def test_the_context_builds_and_renders_each_unlabelled_code_with_no_label(
    driver: Driver,
) -> None:
    sco = prune_to_sco(RetrievalEngine(driver=driver).retrieve(QUESTION))
    text = build_sql_prompt(sco, QUESTION)

    unlabelled = _governed_line(text, UNLABELLED)
    for code in GRIT:
        assert f"'{code}' (no label)" in unlabelled, text
    labelled = _governed_line(text, LABELLED)
    for code, label in FINISH.items():
        assert f"'{code}' ({label})" in labelled, text
