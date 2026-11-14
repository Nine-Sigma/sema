"""Detected joins copy the column references the connector built; they never invent one."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import sema.engine.join_detector as join_detector
import sema.engine.join_detector_utils as join_detector_utils
from sema.connectors.databricks import DatabricksConnector
from sema.engine.join_detector import JoinDetector, to_fk_assertion
from sema.graph.loader_utils import fetch_columns_by_schema
from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.config import Neo4jConfig
from sema.models.physical_key import CanonicalRef
from tests.support.connector_fixture import (
    CATALOG,
    MAIN_SCHEMA,
    fixture_build_config,
    fixture_environment,
)

pytestmark = pytest.mark.unit


def _column_rows() -> list[dict[str, Any]]:
    """Column node properties as the graph holds them after the connector's L1 extraction."""
    databricks = fixture_build_config(Neo4jConfig()).databricks
    with fixture_environment():
        assertions = DatabricksConnector(databricks).extract(CATALOG, [MAIN_SCHEMA])
    return [
        _row(a) for a in assertions if a.predicate == AssertionPredicate.COLUMN_EXISTS
    ]


def _row(assertion: Assertion) -> dict[str, Any]:
    key = CanonicalRef.parse(assertion.subject_ref)
    return {
        "name": key.column,
        "table_name": key.table,
        "catalog": key.catalog_or_db,
        "schema_name": key.schema,
        "data_type": assertion.payload["data_type"],
        "nullable": assertion.payload["nullable"],
        "comment": assertion.payload["comment"],
        "ref": assertion.subject_ref,
    }


def _detected(rows: list[dict[str, Any]]) -> list[Assertion]:
    loader = MagicMock()
    loader._run_read.return_value = rows
    columns = fetch_columns_by_schema(loader, MAIN_SCHEMA)
    fks = JoinDetector().detect(columns=columns, source_schema=MAIN_SCHEMA)
    return [to_fk_assertion(fk, run_id="run-1") for fk in fks]


@pytest.fixture(scope="module")
def column_rows() -> list[dict[str, Any]]:
    return _column_rows()


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    from sema.log import logger

    lines: list[str] = []
    handler_id = logger.add(lambda message: lines.append(str(message)), level="WARNING")
    yield lines
    logger.remove(handler_id)


def test_every_fk_reference_is_a_column_reference_the_connector_built(
    column_rows: list[dict[str, Any]],
) -> None:
    ref_by_column = {(r["table_name"], r["name"]): r["ref"] for r in column_rows}
    detected = _detected(column_rows)

    assert detected
    for assertion in detected:
        payload = assertion.payload
        assert assertion.subject_ref == ref_by_column[(payload["fk_table"], payload["fk_column"])]
        assert assertion.object_ref == ref_by_column[(payload["pk_table"], payload["pk_column"])]


def test_a_column_read_without_a_reference_yields_no_candidate_and_one_warning(
    column_rows: list[dict[str, Any]], warnings: list[str],
) -> None:
    payload = _detected(column_rows)[0].payload
    dropped = (payload["fk_table"], payload["fk_column"])
    rows = [
        {k: v for k, v in r.items() if k != "ref"} if (r["table_name"], r["name"]) == dropped
        else r
        for r in column_rows
    ]

    remaining = _detected(rows)

    assert remaining
    for assertion in remaining:
        p = assertion.payload
        assert dropped not in {(p["fk_table"], p["fk_column"]), (p["pk_table"], p["pk_column"])}
    assert len([w for w in warnings if f"{dropped[0]}.{dropped[1]}" in w]) == 1


def test_no_detector_source_line_holds_a_scheme_literal() -> None:
    for module in (join_detector, join_detector_utils):
        source = Path(str(module.__file__)).read_text().splitlines()
        assert [line for line in source if "://" in line] == [], module.__name__
