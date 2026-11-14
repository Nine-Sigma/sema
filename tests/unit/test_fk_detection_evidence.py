"""Foreign keys are detected from data evidence on the neutral fixture, not from table names."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.connectors.databricks import DatabricksConnector
from sema.engine.join_detector import DEFAULT_SAMPLE_CAP
from sema.engine.join_detector_utils import enumerate_candidates_from_metadata
from sema.graph.loader_utils import fetch_columns_by_schema
from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.config import BuildConfig, Neo4jConfig
from sema.models.physical_key import CanonicalRef
from sema.pipeline.orchestrate_utils import run_fk_detection
from tests.support.connector_fixture import (
    CATALOG,
    DECOY_JOINS,
    EXPECTED_DETECTED_JOINS,
    MAIN_SCHEMA,
    UNDETECTABLE_DECLARED_JOIN,
    FixtureJoin,
    fixture_build_config,
    fixture_environment,
)

pytestmark = pytest.mark.unit

_IN_LIST = re.compile(r"\bIN\s*\((?P<values>[^)]*)\)", re.IGNORECASE)


@dataclass(frozen=True)
class DetectionRun:
    detected: list[Assertion]
    statements: list[str]
    unanswered: list[str]
    candidate_count: int


def _column_row(assertion: Assertion) -> dict[str, Any]:
    key = CanonicalRef.parse(assertion.subject_ref)
    return {
        "name": key.column, "table_name": key.table, "catalog": key.catalog_or_db,
        "schema_name": key.schema, "data_type": assertion.payload["data_type"],
        "nullable": assertion.payload["nullable"], "comment": assertion.payload["comment"],
        "ref": assertion.subject_ref,
    }


@pytest.fixture(scope="module")
def run() -> Iterator[DetectionRun]:
    databricks = fixture_build_config(Neo4jConfig()).databricks
    with fixture_environment() as connections:
        connector = DatabricksConnector(databricks)
        rows = [
            _column_row(a) for a in connector.extract(CATALOG, [MAIN_SCHEMA])
            if a.predicate == AssertionPredicate.COLUMN_EXISTS
        ]
        statements: list[str] = []
        execute = connector.execute_query

        def recording(query: str) -> list[tuple[Any, ...]]:
            statements.append(query)
            return execute(query)

        connector.execute_query = recording  # type: ignore[method-assign]
        extraction_unanswered = {u for c in connections for u in c.unanswered}
        loader = MagicMock()
        loader._run_read.return_value = rows
        config = BuildConfig(catalog=CATALOG, schemas=[MAIN_SCHEMA], enable_fk_detection=True)
        detected = run_fk_detection(loader, connector, config, [MAIN_SCHEMA], run_id="run-1")
    candidates = enumerate_candidates_from_metadata(fetch_columns_by_schema(loader, MAIN_SCHEMA))
    yield DetectionRun(
        detected=detected.get(MAIN_SCHEMA, []),
        statements=statements,
        unanswered=[
            u for c in connections for u in c.unanswered if u not in extraction_unanswered
        ],
        candidate_count=len(candidates),
    )


def _matches(assertion: Assertion, join: FixtureJoin) -> bool:
    payload = assertion.payload
    if (payload["fk_table"], payload["fk_column"]) != (
        join.referencing_table, join.referencing_column,
    ):
        return False
    if join.referenced_table is None:
        return True
    return (payload["pk_table"], payload["pk_column"]) == (
        join.referenced_table, join.referenced_column,
    )


def _containment_statements(statements: list[str]) -> list[str]:
    return [s for s in statements if _IN_LIST.search(s)]


def test_every_expected_join_is_detected_above_the_default_threshold(run: DetectionRun) -> None:
    missing = [
        join for join in EXPECTED_DETECTED_JOINS
        if not any(_matches(a, join) for a in run.detected)
    ]
    assert missing == []
    assert all(a.predicate == AssertionPredicate.FK_TO for a in run.detected)
    assert all(a.confidence >= 0.80 for a in run.detected)


def test_no_decoy_join_is_detected(run: DetectionRun) -> None:
    found = [
        (join, a.payload) for join in DECOY_JOINS for a in run.detected if _matches(a, join)
    ]
    assert found == []


def test_the_declared_key_with_no_name_affinity_is_not_detected(run: DetectionRun) -> None:
    assert [a.payload for a in run.detected if _matches(a, UNDETECTABLE_DECLARED_JOIN)] == []


def test_containment_is_checked_with_at_most_one_bounded_statement_per_candidate(
    run: DetectionRun,
) -> None:
    containment = _containment_statements(run.statements)
    assert containment, "no containment statement reached the warehouse"
    assert len(containment) <= run.candidate_count
    for statement in containment:
        match = _IN_LIST.search(statement)
        assert match is not None
        assert len(match["values"].split(",")) <= DEFAULT_SAMPLE_CAP


def test_every_statement_is_answered_and_none_is_an_unbounded_scan(run: DetectionRun) -> None:
    assert run.unanswered == []
    assert [s for s in run.statements if "NOT IN" in s.upper()] == []
