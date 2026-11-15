"""Shared setup for test_sema_entity_lint.py and the tests that reuse it."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import duckdb

from sema.compile.entity_identity import identity_cte
from sema.models.target.identity import EntityIdentity
from showcase.ontology_query_benchmark.sema_entity_utils import ExposedEntity

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
