from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from sema.models.assertions import (
    Assertion,
    AssertionStatus,
)

PROVIDER_ENV_PREFIXES = (
    "EMBEDDING_",
    "LLM_",
    "OPENROUTER_",
    "INTAKE_MODEL_",
    "INTAKE_TRACING_",
    "DATABRICKS_",
    "MLFLOW_",
    "LANGCHAIN_",
    "LANGSMITH_",
)


@pytest.fixture(autouse=True)
def no_provider_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """A unit test sets every provider variable it needs; the shell's and a .env's never apply."""
    if request.node.get_closest_marker("unit") is None:
        return
    for name in [name for name in os.environ if name.startswith(PROVIDER_ENV_PREFIXES)]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    # Without a host the Databricks SDK falls back to ~/.databrickscfg profiles.
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", os.devnull)


@pytest.fixture
def neo4j_driver():
    """Real Neo4j driver for integration tests. Requires running Neo4j."""
    neo4j = pytest.importorskip("neo4j")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "graphrag")
    driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
    yield driver
    driver.close()


@pytest.fixture
def clean_neo4j(neo4j_driver):
    """Wipe Neo4j before each integration test."""
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    yield neo4j_driver
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


@pytest.fixture
def mock_llm():
    """Mock LLM that returns configurable responses."""
    llm = MagicMock()
    llm.invoke = MagicMock(return_value="")
    return llm


def make_assertion(
    subject_ref: str,
    predicate: str,
    value: str | dict | None = None,
    object_ref: str | None = None,
    source: str = "test",
    confidence: float = 0.9,
    status: AssertionStatus = AssertionStatus.AUTO,
    run_id: str | None = None,
) -> Assertion:
    """Factory for creating test assertions."""
    return Assertion(
        id=str(uuid.uuid4()),
        subject_ref=subject_ref,
        predicate=predicate,
        payload=value if isinstance(value, dict) else {"value": value} if value is not None else {},
        object_ref=object_ref,
        source=source,
        confidence=confidence,
        status=status,
        run_id=run_id or str(uuid.uuid4()),
        observed_at=datetime.now(timezone.utc),
    )
