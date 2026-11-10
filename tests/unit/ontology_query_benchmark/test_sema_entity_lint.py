"""Arm S rejects a multi-column key count as a repairable error naming the entity key.

``COUNT(DISTINCT a, b)`` skips every row with a NULL key part. When a statement reads
an exposed entity's source relation that way, the S executor returns a ``sql_error``
that tells the model to count ``<entity>_key`` from the entity instead.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import duckdb
import pytest

from sema.compile.entity_identity import identity_cte
from sema.models.target.identity import EntityIdentity
from showcase.ontology_query_benchmark.sema_entity_utils import EntityExecutor, ExposedEntity

pytestmark = pytest.mark.unit

VISIT_ROWS = [("v1", "a"), ("v1", "a"), ("v1", "b"), ("v1", None), (None, "a")]
LEGAL_ROWS = [("D1", "2025-06-30"), ("D2", "2025-06-30"), ("D3", None)]


class DuckWarehouse:
    def __init__(self, harness: SimpleNamespace) -> None:
        self.harness, self.sqls = harness, []
        self.con = duckdb.connect()
        self.con.execute("ATTACH ':memory:' AS cat")
        self.con.execute("CREATE SCHEMA cat.sch")
        for table in ("visits", "other_visits"):
            self.con.execute(f"CREATE TABLE cat.sch.{table} (visit_id VARCHAR, site VARCHAR)")
            self.con.executemany(f"INSERT INTO cat.sch.{table} VALUES (?, ?)", VISIT_ROWS)
        self.con.execute(
            'CREATE TABLE cat.sch.legals ("DOCUMENT ID" VARCHAR, "GOOD THROUGH DATE" VARCHAR)'
        )
        self.con.executemany("INSERT INTO cat.sch.legals VALUES (?, ?)", LEGAL_ROWS)

    def execute(self, sql: str, row_limit: int) -> Any:
        self.sqls.append(sql)
        models = self.harness.models
        try:
            cursor = self.con.execute(sql)
        except duckdb.Error as exc:
            return models.Execution(False, (), (), 0.0, "sql_error", str(exc)[:200])
        return models.Execution(True, (), tuple(tuple(r) for r in cursor.fetchmany(row_limit)), 0.0)


def _entity(name: str, relation: str, keys: list[str]) -> ExposedEntity:
    identity = EntityIdentity(key_columns=tuple(keys))
    return ExposedEntity(name, relation, identity, identity_cte(name, relation, identity, "duckdb"))


ENTITIES = (
    _entity("visit", "cat.sch.visits", ["visit_id", "site"]),
    _entity("document", "cat.sch.legals", ["DOCUMENT ID", "GOOD THROUGH DATE"]),
)


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
