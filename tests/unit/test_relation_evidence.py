"""Done-when 1 (N3): count evidence on a loaded relation, and the declared-key read.

Every count runs on the identity fixture through the DuckDB connector and is compared with
the number the fixture's rows give. A tuple with a NULL part counts as a value. Each count is
one aggregate statement through the connector's timed read, and a statement past its timeout
gives no count. No model is called.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from sema.connectors.duckdb import DuckDBConnector
from tests.support.connector_fixture_identity import (
    IDENTITY_CATALOG,
    IDENTITY_SCHEMA,
    IDENTITY_TABLES,
    TABLES,
    conflict_counts,
    contained_counts,
    distinct_count,
    non_key_columns,
    null_row_count,
    write_identity_database,
)
from tests.support.connector_fixture_identity_runs import fixture_connector

pytestmark = pytest.mark.unit


def _evidence_type() -> Any:
    return importlib.import_module("sema.engine.warehouse_lookup").RelationEvidence


@pytest.fixture(scope="module")
def database(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_identity_database(tmp_path_factory.mktemp("evidence"))


@pytest.fixture
def connector(database: Path) -> Iterator[DuckDBConnector]:
    opened = fixture_connector(database)
    yield opened
    opened.close()


def _relation(table: str) -> tuple[str, str]:
    return (IDENTITY_SCHEMA, table)


def _evidence(connector: DuckDBConnector) -> Any:
    return _evidence_type()(connector, IDENTITY_CATALOG)


@pytest.mark.parametrize("table", [t.name for t in IDENTITY_TABLES])
def test_rows_distinct_and_null_rows_equal_the_fixture(
    connector: DuckDBConnector, table: str,
) -> None:
    evidence = _evidence(connector)
    columns = TABLES[table].column_names
    assert evidence.rows(_relation(table)) == len(TABLES[table].rows)
    for picked in (columns[:1], columns[:2], columns):
        assert evidence.distinct(_relation(table), picked) == distinct_count(table, picked)
        assert evidence.null_rows(_relation(table), picked) == null_row_count(table, picked)


def test_a_tuple_with_a_null_part_is_counted(connector: DuckDBConnector) -> None:
    evidence = _evidence(connector)
    key = ["pier_code", "slot_no"]

    assert evidence.distinct(_relation("mooring"), key) == len(TABLES["mooring"].rows) == 6
    assert evidence.null_rows(_relation("mooring"), key) == 3
    assert evidence.distinct(_relation("crane_rev"), ["crane_no"]) == 3


@pytest.mark.parametrize(("table", "key"), [
    ("hull_rev", ["hull_no"]), ("crane_rev", ["crane_no"]), ("buoy_check", ["buoy_no"]),
    ("berth_call", ["vessel_no"]), ("berth", ["berth_no"]),
])
def test_conflicts_count_key_groups_and_each_attributes_conflicting_groups(
    connector: DuckDBConnector, table: str, key: list[str],
) -> None:
    attributes = list(non_key_columns(table, key))

    measured = _evidence(connector).conflicts(_relation(table), key, attributes)

    groups, conflicting = measured
    assert (groups, dict(conflicting)) == conflict_counts(table, key, attributes)


def test_a_changing_attribute_conflicts_and_a_fixed_one_does_not(
    connector: DuckDBConnector,
) -> None:
    measured = _evidence(connector).conflicts(
        _relation("hull_rev"), ["hull_no"], ["paint_band", "keel_class"])

    assert measured is not None
    assert (measured[0], dict(measured[1])) == (4, {"paint_band": 2, "keel_class": 0})


@pytest.mark.parametrize(("table", "owner"), [
    ("hull_survey", "hull_rev"), ("hull_cargo", "hull_rev"), ("hull_quote", "hull_rev"),
    ("valve", "pump"), ("pump", "valve"),
])
def test_contained_counts_distinct_non_null_tuples_and_those_in_the_owner(
    connector: DuckDBConnector, table: str, owner: str,
) -> None:
    column = "hull_no" if owner == "hull_rev" else "unit_no"

    measured = _evidence(connector).contained(
        _relation(table), [column], _relation(owner), [column])

    assert tuple(measured) == contained_counts(table, [column], owner, [column])


def test_the_declared_key_read_returns_the_fixtures_declared_keys(
    connector: DuckDBConnector,
) -> None:
    read = connector.extract_unique_keys(IDENTITY_CATALOG, IDENTITY_SCHEMA)

    found = {(str(table).lower(), tuple(sorted(columns))) for table, columns in read}
    declared = {(t.name, tuple(sorted(k))) for t in IDENTITY_TABLES
                for k in (t.primary_key, *t.unique) if k}
    assert found == declared
    assert ("mooring", ("pier_code", "slot_no")) in found


def test_the_scorecard_reads_declared_keys_through_the_same_read(
    connector: DuckDBConnector,
) -> None:
    scorecard = importlib.import_module("sema.eval.discovery_scorecard")
    calls: list[tuple[str, str]] = []
    reads = connector.extract_unique_keys

    def spy(catalog: str, schema: str) -> Any:
        calls.append((catalog, schema))
        return reads(catalog, schema)

    with patch.object(connector, "extract_unique_keys", spy):
        cells = scorecard.declared_constraints(connector, IDENTITY_CATALOG, [IDENTITY_SCHEMA])

    assert calls == [(IDENTITY_CATALOG, IDENTITY_SCHEMA)]
    assert ("hull_rev", "hull_no+rev_no") in cells["candidate_key"]


class _Spy:
    """Records each statement's bounds and its result rows."""

    def __init__(self, connector: DuckDBConnector) -> None:
        self.timed: list[tuple[str, Any, int]] = []
        self.plain: list[str] = []
        self._timed, self._plain = connector.execute_query_with_columns, connector.execute_query

    def execute_query_with_columns(self, sql: str, **bounds: Any) -> Any:
        names, rows = self._timed(sql, **bounds)
        self.timed.append((sql, bounds.get("timeout_s"), len(rows)))
        return names, rows

    def execute_query(self, sql: str) -> list[tuple[Any, ...]]:
        self.plain.append(sql)
        return self._plain(sql)


def test_each_count_is_one_timed_aggregate_statement(connector: DuckDBConnector) -> None:
    spy = _Spy(connector)
    with patch.object(connector, "execute_query_with_columns", spy.execute_query_with_columns), \
            patch.object(connector, "execute_query", spy.execute_query):
        evidence = _evidence(connector)
        evidence.rows(_relation("hull_rev"))
        evidence.distinct(_relation("hull_rev"), ["hull_no"])
        evidence.null_rows(_relation("crane_rev"), ["crane_no"])
        evidence.conflicts(_relation("hull_rev"), ["hull_no"], ["paint_band", "keel_class"])
        evidence.contained(_relation("hull_survey"), ["hull_no"], _relation("hull_rev"),
                           ["hull_no"])

    assert spy.plain == []
    assert len(spy.timed) == 5
    assert all(timeout is not None and timeout > 0 for _, timeout, _ in spy.timed)
    assert all(rows == 1 for _, _, rows in spy.timed), "a count statement returns one row"


def test_a_statement_past_its_timeout_gives_no_count(database: Path) -> None:
    connector = fixture_connector(database, lambda sql: "crane_rev" in sql)
    try:
        evidence = _evidence(connector)
        assert evidence.rows(_relation("crane_rev")) is None
        assert evidence.distinct(_relation("crane_rev"), ["crane_no"]) is None
        assert evidence.rows(_relation("hull_rev")) == len(TABLES["hull_rev"].rows)
    finally:
        connector.close()


def test_no_model_is_called(connector: DuckDBConnector) -> None:
    with patch("sema.llm_client.LLMClient.invoke", side_effect=AssertionError("model called")):
        evidence = _evidence(connector)
        assert evidence.rows(_relation("berth")) == len(TABLES["berth"].rows)
        assert evidence.conflicts(_relation("hull_rev"), ["hull_no"], ["paint_band"]) is not None
