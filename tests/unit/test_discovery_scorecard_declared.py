"""Done-when 6 (N7): the scorecard rechecks a join on several columns and counts declared
joins apart from discovered ones.

The recheck reads the declared-join fixture in DuckDB through the DuckDB connector. Each join
is given as the text a `JoinPath` holds in `join_predicates`: the referenced relation on the
left, one predicate per column pair, in pair order. The graph reads run on a stand-in driver
that returns only the columns the statement's `RETURN` names.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sema.connectors.duckdb import DuckDBConnector
from sema.eval.discovery_scorecard import facts_from_graph, recheck, render, score_discovery
from sema.eval.discovery_scorecard_utils import graph_join_key
from sema.models.config import DuckDBConfig
from tests.support.connector_fixture_declared import write_declared_database
from tests.support.connector_fixture_identity import IDENTITY_CATALOG, IDENTITY_SCHEMA

pytestmark = pytest.mark.unit

DECLARED, DISCOVERED = "joins from declarations: ", "joins discovered: "


def predicates(referenced: str, referencing: str, *columns: str) -> str:
    return json.dumps([
        {"left_table": referenced, "left_column": column, "right_table": referencing,
         "right_column": column, "operator": "="}
        for column in columns
    ])


HELD = predicates("crate_head", "crate_line", "crate_no", "seal_day")
REPEATED = predicates("bale_head", "bale_line", "bale_no", "weigh_day")
ORPHANED = predicates("crate_head_spare", "crate_line", "crate_no", "seal_day")


@pytest.fixture
def connector(tmp_path: Path) -> Iterator[DuckDBConnector]:
    opened = DuckDBConnector(DuckDBConfig(path=str(write_declared_database(tmp_path))))
    yield opened
    opened.close()


def _produced(*texts: str) -> dict[str, dict[tuple[str, ...], str]]:
    return {"join": {graph_join_key(text): "key_only" for text in texts}}


def _rechecked(connector: DuckDBConnector, *texts: str) -> int:
    return recheck(connector, IDENTITY_CATALOG, [IDENTITY_SCHEMA], _produced(*texts),
                   join_predicates=list(texts))["join"]


def test_a_join_on_two_columns_whose_pairs_hold_rechecks(connector: DuckDBConnector) -> None:
    assert _rechecked(connector, HELD) == 1


def test_a_join_onto_a_repeated_referenced_tuple_fails_the_recheck(
    connector: DuckDBConnector,
) -> None:
    assert _rechecked(connector, REPEATED) == 0
    assert _rechecked(connector, HELD, REPEATED) == 1


def test_a_join_whose_pairs_are_orphans_above_the_floor_fails_the_recheck(
    connector: DuckDBConnector,
) -> None:
    assert _rechecked(connector, ORPHANED) == 0
    assert _rechecked(connector, HELD, ORPHANED) == 1


def test_a_recheck_given_no_predicates_keeps_todays_result(connector: DuckDBConnector) -> None:
    assert recheck(connector, IDENTITY_CATALOG, [IDENTITY_SCHEMA], _produced(HELD))["join"] == 0


def _card() -> Any:
    produced = _produced(HELD)
    return score_discovery(produced, produced, {"join": 1})


def test_the_report_prints_declared_and_discovered_joins_under_the_table() -> None:
    table = render(_card())

    text = render(_card(), declared=3, discovered=2)

    assert text.startswith(table)
    lines = text[len(table):].splitlines()
    assert f"{DECLARED}3" in lines and f"{DISCOVERED}2" in lines


def test_a_render_given_no_counts_keeps_todays_table() -> None:
    text = render(_card())

    assert DECLARED not in text and DISCOVERED not in text
    assert all(line.startswith("|") for line in text.splitlines())


class _Session:
    """Answers a statement with the stored rows, keeping only the aliases its RETURN names."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def run(self, statement: str, **_: Any) -> list[dict[str, Any]]:
        returned = re.findall(r"\bAS (\w+)", statement.split("RETURN", 1)[1])
        return [{name: row[name] for name in returned} for row in self.rows]


class _Driver:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def session(self) -> _Session:
        return _Session(self.rows)


GRAPH_ROWS: list[dict[str, Any]] = [
    {"predicates": HELD, "cardinality": None, "join_source": "dictionary_declared"},
    {"predicates": predicates("crate_head", "crate_line", "crate_no"),
     "cardinality": "many_to_one", "join_source": "detected"},
]


def test_joins_from_graph_returns_each_joins_predicates_and_source() -> None:
    from sema.eval.discovery_scorecard import joins_from_graph

    joins_read = joins_from_graph(_Driver(GRAPH_ROWS), IDENTITY_CATALOG, [IDENTITY_SCHEMA])

    assert [tuple(pair) for pair in joins_read] == [
        (row["predicates"], row["join_source"]) for row in GRAPH_ROWS]


def test_facts_from_graph_keeps_its_result() -> None:
    facts = facts_from_graph(_Driver(GRAPH_ROWS), IDENTITY_CATALOG, [IDENTITY_SCHEMA])

    assert facts == {"join": {
        graph_join_key(GRAPH_ROWS[0]["predicates"]): "key_only",
        graph_join_key(GRAPH_ROWS[1]["predicates"]): "many_to_one",
    }}
