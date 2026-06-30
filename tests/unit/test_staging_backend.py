"""US-013: the multi-warehouse staging backend (DuckDB + Databricks).

Hermetic: a fake cursor records every statement so we can assert the rendered
SQL per dialect without a live warehouse. The DuckDB backend must reproduce the
existing temp-build + scoped-swap behaviour; the Databricks backend must emit an
atomic Delta ``INSERT ... REPLACE WHERE`` scoped on (source_schema, source_table)
and render the compiled SELECT in the databricks dialect.
"""

from __future__ import annotations

from typing import Any, Sequence

import duckdb
import pytest
import sqlglot

from sema.compile.compiler_utils import (
    CompileContext,
    SourceTableSpec,
    StagingColumns,
    StagingDecision,
    build_staging_select,
)
from sema.compile.staging_backend import (
    DATABRICKS_BACKEND,
    DUCKDB_BACKEND,
    staging_backend_for,
)

pytestmark = pytest.mark.unit


_COLS = StagingColumns(
    source_value_column="source_code",
    target_concept_column="target_value",
)
_SOURCE = SourceTableSpec(schema="study", table="sample", value_column="CODE")
_CONTEXT = CompileContext(
    resolver_policy_ref="policy.x", vocab_release="rel-1", run_id="run-1"
)
_DECISIONS = (
    StagingDecision("LUAD", 4314337, "RESOLVED", None, "auto_accepted"),
    StagingDecision("ZZZZ", None, "NO_MAP", "dead end", "auto_accepted"),
)


def _select() -> Any:
    return build_staging_select(_COLS, _SOURCE, _CONTEXT, _DECISIONS)


class _FakeCursor:
    """Records executed SQL; returns a canned count for the final fetch."""

    def __init__(self, count: int = 7) -> None:
        self.statements: list[str] = []
        self.params: list[Sequence[Any] | None] = []
        self._count = count

    def execute(self, sql: str, parameters: Sequence[Any] | None = None) -> "_FakeCursor":
        self.statements.append(sql)
        self.params.append(parameters)
        return self

    def fetchone(self) -> tuple[int]:
        return (self._count,)

    def fetchall(self) -> list[Any]:
        return [(self._count,)]


def test_dialects_and_factory() -> None:
    assert DUCKDB_BACKEND.dialect == "duckdb"
    assert DATABRICKS_BACKEND.dialect == "databricks"
    assert staging_backend_for("duckdb") is DUCKDB_BACKEND
    assert staging_backend_for("databricks") is DATABRICKS_BACKEND


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError):
        staging_backend_for("postgres")


def test_databricks_quoting_and_scope() -> None:
    assert DATABRICKS_BACKEND.qualified("s", "t") == "`s`.`t`"
    pred, params = DATABRICKS_BACKEND.scope_predicate(_COLS, "study", "sample")
    assert "`source_schema` = 'study'" in pred
    assert "`source_table` = 'sample'" in pred
    assert params == []  # literals inlined, no bound params


def test_duckdb_scope_uses_bound_params() -> None:
    pred, params = DUCKDB_BACKEND.scope_predicate(_COLS, "study", "sample")
    assert pred == '"source_schema" = ? AND "source_table" = ?'
    assert params == ["study", "sample"]


def test_databricks_write_emits_atomic_replace_where() -> None:
    cur = _FakeCursor(count=5)
    written = DATABRICKS_BACKEND.write_staging(
        cur,
        _select(),
        columns=_COLS,
        source=_SOURCE,
        staging_schema="sema_staging",
        staging_table="condition_staging",
    )
    assert written == 5
    joined = "\n".join(cur.statements)
    assert "CREATE SCHEMA IF NOT EXISTS `sema_staging`" in joined
    assert "USING DELTA" in joined
    # Atomic, idempotent, scoped replace — no temp table, no BEGIN/COMMIT.
    assert "REPLACE WHERE" in joined
    assert "TEMP TABLE" not in joined
    assert "BEGIN TRANSACTION" not in joined
    # No FK closure / person column leaked into the staging projection.
    assert "person_id" not in joined


def test_databricks_statements_parse_in_databricks_dialect() -> None:
    cur = _FakeCursor()
    DATABRICKS_BACKEND.write_staging(
        cur,
        _select(),
        columns=_COLS,
        source=_SOURCE,
        staging_schema="sema_staging",
        staging_table="condition_staging",
    )
    for stmt in cur.statements:
        # Every emitted statement must be valid Databricks SQL.
        sqlglot.parse_one(stmt, dialect="databricks")


def test_duckdb_write_round_trips_against_real_duckdb(tmp_path: Any) -> None:
    conn = duckdb.connect(str(tmp_path / "x.duckdb"))
    conn.execute('CREATE SCHEMA "study"')
    conn.execute('CREATE TABLE "study"."sample" (CODE VARCHAR)')
    conn.executemany('INSERT INTO "study"."sample" VALUES (?)', [("LUAD",), ("ZZZZ",)])
    written = DUCKDB_BACKEND.write_staging(
        conn,
        _select(),
        columns=_COLS,
        source=_SOURCE,
        staging_schema="sema_staging",
        staging_table="condition_staging",
    )
    assert written == 2
    rows = conn.execute(
        'SELECT source_code, target_value FROM "sema_staging"."condition_staging" '
        "ORDER BY source_code"
    ).fetchall()
    assert rows == [("LUAD", 4314337), ("ZZZZ", None)]
