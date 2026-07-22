"""S1-08 — registry→Databricks bridge (hermetic).

A fake cursor records the emitted SQL so we can assert the Delta DDL + batched
VALUES without a live warehouse, and that every statement parses in the databricks
dialect. The rows come from a real DuckDB registry so the bridge is exercised over
the actual frozen contract, not a hand-built stand-in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import duckdb
import pytest
import sqlglot

from sema.resolve.identity_bridge import (
    bridge_identity_registry,
    databricks_create_registry_table_sql,
)
from sema.resolve.identity_registry import IdentityRegistry
from sema.resolve.identity_registry_utils import FROZEN_COLUMNS

pytestmark = pytest.mark.unit


class _FakeCursor:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, sql: str, parameters: Sequence[Any] | None = None) -> "_FakeCursor":
        self.statements.append(sql)
        return self


def _registry(tmp_path: Path, keys: list[str]) -> IdentityRegistry:
    reg = IdentityRegistry(duckdb.connect(str(tmp_path / "poc.duckdb")))
    reg.get_or_create([("cbio_msk", k) for k in keys], run_id="s1-08")
    return reg


def test_create_table_is_delta_over_frozen_columns() -> None:
    sql = databricks_create_registry_table_sql("sema_identity", "entity_identity")
    assert "CREATE OR REPLACE TABLE `sema_identity`.`entity_identity`" in sql
    assert "USING DELTA" in sql
    for col in FROZEN_COLUMNS:
        assert f"`{col}`" in sql
    assert "`entity_id` BIGINT" in sql


def test_bridge_emits_create_then_batched_inserts(tmp_path: Path) -> None:
    reg = _registry(tmp_path, ["P-1", "P-2", "P-3"])
    cur = _FakeCursor()
    written = bridge_identity_registry(
        cur, reg.read_all(), schema="sema_identity", table="entity_identity",
        batch_rows=2,
    )
    assert written == 3
    joined = "\n".join(cur.statements)
    assert "CREATE SCHEMA IF NOT EXISTS `sema_identity`" in cur.statements[0]
    assert "CREATE OR REPLACE TABLE" in cur.statements[1]
    inserts = [s for s in cur.statements if s.startswith("INSERT INTO")]
    assert len(inserts) == 2  # 3 rows, batch of 2 -> two INSERTs
    # entity_id inlined unquoted; the P-* keys quoted.
    assert "'P-1'" in joined
    assert "'cbio_msk'" in joined


def test_all_statements_parse_in_databricks_dialect(tmp_path: Path) -> None:
    reg = _registry(tmp_path, ["P-1", "P-2"])
    cur = _FakeCursor()
    bridge_identity_registry(cur, reg.read_all())
    for stmt in cur.statements:
        sqlglot.parse_one(stmt, dialect="databricks")


def test_single_quote_in_key_is_escaped(tmp_path: Path) -> None:
    reg = _registry(tmp_path, ["O'Brien-1"])
    cur = _FakeCursor()
    bridge_identity_registry(cur, reg.read_all())
    insert = next(s for s in cur.statements if s.startswith("INSERT INTO"))
    assert "'O''Brien-1'" in insert
    sqlglot.parse_one(insert, dialect="databricks")
