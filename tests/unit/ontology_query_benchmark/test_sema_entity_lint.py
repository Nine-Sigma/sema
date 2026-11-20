"""Arm S rejects a multi-column key count as a repairable error naming the entity key.

``COUNT(DISTINCT a, b)`` skips every row with a NULL key part. When a statement reads
an exposed entity's source relation that way, the S executor returns a ``sql_error``
that tells the model to count ``<entity>_key`` from the entity instead.
"""

from __future__ import annotations

from types import SimpleNamespace

import duckdb
import pytest

from showcase.ontology_query_benchmark.sema_entity_utils import EntityExecutor
from tests.unit.ontology_query_benchmark.sema_entity_lint_env import ENTITIES, DuckWarehouse

pytestmark = pytest.mark.unit


def _executor(harness: SimpleNamespace) -> tuple[EntityExecutor, DuckWarehouse]:
    warehouse = DuckWarehouse(harness)
    return EntityExecutor(harness, warehouse, ENTITIES, "duckdb"), warehouse


@pytest.mark.parametrize(("sql", "key"), [
    ("SELECT COUNT(DISTINCT visit_id, site) AS n FROM cat.sch.VISITS", "visit_key"),
    ('SELECT COUNT(DISTINCT "DOCUMENT ID", "GOOD THROUGH DATE") AS n FROM cat.sch.legals',
     "document_key"),
])  # fmt: skip
def test_a_key_part_count_on_an_entity_relation_is_a_repairable_error_naming_the_key(
    harness: SimpleNamespace, sql: str, key: str
) -> None:
    executor, warehouse = _executor(harness)
    execution = executor.execute(sql, 1000)
    assert (execution.ok, execution.error_class) == (False, "sql_error")
    assert key in execution.message
    assert "is NULL" in execution.message
    assert warehouse.sqls == []


def test_counting_the_entity_key_from_the_entity_runs(harness: SimpleNamespace) -> None:
    executor, _ = _executor(harness)
    execution = executor.execute("SELECT COUNT(DISTINCT visit_key) AS n FROM visit", 1000)
    assert execution.ok
    assert execution.rows == ((4,),)


def test_a_key_part_count_on_an_unrelated_relation_sharing_key_names_reaches_the_warehouse(
    harness: SimpleNamespace,
) -> None:
    executor, warehouse = _executor(harness)
    sql = "SELECT COUNT(DISTINCT visit_id, site) AS n FROM cat.sch.other_visits"
    executor.execute(sql, 1000)
    assert warehouse.sqls == [sql]
