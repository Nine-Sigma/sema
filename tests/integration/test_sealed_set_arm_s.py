"""Done-when 6: arm S on the fixture graph through the core query path.

The neutral fixture is built through the DuckDB connector (model mocked, `skip_embeddings`),
so the graph holds no vector index. Arm S's provider is
`showcase.ontology_query_benchmark.core_s_provider.CoreSContextProvider(driver, embedder)`,
an `SProvider`: called with `(question, is_denied)` it returns an `SContextLike` (`text`,
`identity`, `findings`) from `RetrievalEngine.retrieve` -> `prune_to_sco` -> the core section
renderer, without the core prompt frame or the question. These tests pass `embedder=None`,
so retrieval reaches the context through the lexical path alone: the question's tokens
match entity and property names in the graph. The mocked statement then executes through
the DuckDB connector inside `sealed_set.run_sealed_set`, and the attempt is scored.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from sema.models.config import Neo4jConfig
from sema.pipeline.orchestrate import run_build
from showcase.ontology_query_benchmark import runner_utils
from showcase.ontology_query_benchmark.profile import load_profile
from tests.support.connector_fixture import (
    MAIN_SCHEMA,
    fake_model_environment,
    fixture_build_config,
)
from tests.unit.ontology_query_benchmark.sealed_set_env import (
    ARM_REPLIES,
    TOY_PROFILE,
    ScriptedModel,
    freeze_toy_harness,
    materialize,
    question_texts,
    read_jsonl,
)

pytestmark = pytest.mark.integration

QUESTION = "How many customer records are there?"
CUSTOMER_TABLE, CUSTOMER_COLUMN = "fixcat.shop.customer", "customer_name"
CORE_FRAME = "You are a SQL expert"


def _module(name: str) -> ModuleType:
    return importlib.import_module(f"showcase.ontology_query_benchmark.{name}")


def _neo4j_config() -> Neo4jConfig:
    return Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=SecretStr(os.getenv("NEO4J_PASSWORD", "graphrag")),
    )


@pytest.fixture
def frozen(tmp_path: Path) -> Path:
    return freeze_toy_harness(tmp_path / "frozen")


@pytest.fixture
def harness(frozen: Path) -> SimpleNamespace:
    return runner_utils.load_harness(frozen, load_profile(TOY_PROFILE).harness)


@pytest.fixture
def profile_path(tmp_path: Path, frozen: Path) -> Path:
    return materialize(tmp_path / "sealed_set", frozen)


@pytest.fixture
def built(clean_neo4j: Any, profile_path: Path) -> Any:
    database = profile_path.parent / "fixcat.duckdb"
    config = fixture_build_config(_neo4j_config(), database=database).model_copy(
        update={"schemas": [MAIN_SCHEMA]},
    )
    assert config.skip_embeddings is True
    with fake_model_environment():
        report = run_build(config)
    assert report["tables_failed"] == 0, report["failed_tables"]
    return clean_neo4j


def test_the_core_path_gives_arm_s_a_context_without_the_core_frame(built: Any) -> None:
    provider = _module("core_s_provider").CoreSContextProvider(built, embedder=None)

    context = provider(QUESTION, lambda _name: False)

    assert CUSTOMER_TABLE in context.text and CUSTOMER_COLUMN in context.text
    assert CORE_FRAME not in context.text and QUESTION not in context.text
    assert context.identity


def test_a_denied_column_never_reaches_the_core_context(built: Any) -> None:
    provider = _module("core_s_provider").CoreSContextProvider(built, embedder=None)

    context = provider(QUESTION, lambda name: name.lower() == CUSTOMER_COLUMN)

    assert CUSTOMER_TABLE in context.text and CUSTOMER_COLUMN not in context.text


def test_arm_s_executes_the_mocked_statement_through_the_connector_and_is_scored(
    built: Any, harness: SimpleNamespace, profile_path: Path, tmp_path: Path,
) -> None:
    set_dir = profile_path.parent / "neutral"
    provider = _module("core_s_provider").CoreSContextProvider(built, embedder=None)
    model = ScriptedModel(harness, question_texts(set_dir), arm="S")

    _module("sealed_set").run_sealed_set(profile_path, ("S",), 1, model=model,
                                         s_provider=provider, report=tmp_path / "report.md")

    paths = (set_dir / "sealed").rglob("receipts.jsonl")
    receipts = [r for path in paths for r in read_jsonl(path)]
    assert {r["slot_id"]: (r["outcome"], r["strict"]) for r in receipts} == {
        f"S:{qid}:1": ("correct", True) if replies["S"].strict else ("wrong", False)
        for qid, replies in ARM_REPLIES.items()
    }
    assert {(r["generation_calls"], r["sql_executions"]) for r in receipts} == {(1, 1)}
    assert all(r["context_identity"] for r in receipts)
    assert (tmp_path / "report.md").read_text().strip()
