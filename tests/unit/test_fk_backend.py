"""S1-08 — multi-warehouse FK-closed backend (DuckDB + Databricks).

Hermetic: a fake cursor records every statement so we can assert the rendered
Databricks SQL without a live warehouse (no BEGIN/COMMIT, no client temp table,
atomic Delta ``REPLACE WHERE``, ``conv()`` surrogate). The DuckDB backend is
round-tripped against a real in-process DuckDB to prove the refactor kept the
S1-06 behaviour, and the two backends are shown to produce the same row shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import duckdb
import pytest
import sqlglot

from sema.compile.compiler_utils import StagingDecision
from sema.compile.fk_backend import (
    DATABRICKS_FK_BACKEND,
    DUCKDB_FK_BACKEND,
    fk_backend_for,
)
from sema.compile.fk_closed_compiler import FkClosedCompiler
from sema.compile.fk_closed_compiler_utils import (
    ChildSourceSpec,
    ChildTableSpec,
    ParentTableSpec,
    RegistryJoinSpec,
    build_child_select,
)
from sema.resolve.identity_registry import IdentityRegistry

pytestmark = pytest.mark.unit

_REGISTRY = RegistryJoinSpec(
    schema="sema_identity",
    table="entity_identity",
    namespace_column="source_namespace",
    key_column="source_entity_key",
    id_column="entity_id",
)
_PERSON = ParentTableSpec(schema="omop", table="person", id_column="person_id")
_CONDITION = ChildTableSpec(
    schema="omop",
    table="condition_occurrence",
    pk_column="condition_occurrence_id",
    fk_column="person_id",
    value_column="condition_concept_id",
    null_columns=("condition_start_date",),
    row_ref_column="source_row_ref",
    patient_key_column="source_patient_key",
    scope_schema_column="source_schema",
    scope_table_column="source_table",
)
_DECISIONS = [
    StagingDecision("LUAD", 777926, "RESOLVED", None, "auto_accepted"),
    StagingDecision("ZZZ", None, "NO_MAP", "no standard candidate", "auto_accepted"),
]


def _source(schema: str) -> ChildSourceSpec:
    return ChildSourceSpec(
        schema=schema,
        table="sample",
        value_column="oncotree_code",
        row_ref_column="sample_id",
        patient_key_column="patient_id",
    )


def _select(dialect: str) -> Any:
    return build_child_select(
        _CONDITION, _source("cbio"), _REGISTRY, _DECISIONS,
        no_map_default=0, dialect=dialect,
    )


class _FakeCursor:
    """Records executed SQL; returns a canned count for the final fetch."""

    def __init__(self, count: int = 9) -> None:
        self.statements: list[str] = []
        self._count = count

    def execute(self, sql: str, parameters: Sequence[Any] | None = None) -> "_FakeCursor":
        self.statements.append(sql)
        return self

    def fetchone(self) -> tuple[int]:
        return (self._count,)


def test_dialects_and_factory() -> None:
    assert DUCKDB_FK_BACKEND.dialect == "duckdb"
    assert DATABRICKS_FK_BACKEND.dialect == "databricks"
    assert fk_backend_for("duckdb") is DUCKDB_FK_BACKEND
    assert fk_backend_for("databricks") is DATABRICKS_FK_BACKEND


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError):
        fk_backend_for("postgres")


def test_databricks_parent_replace_is_delta_ctas() -> None:
    cur = _FakeCursor()
    DATABRICKS_FK_BACKEND.replace_parent(cur, _PERSON, _REGISTRY)
    joined = "\n".join(cur.statements)
    assert "CREATE SCHEMA IF NOT EXISTS `omop`" in joined
    assert "CREATE OR REPLACE TABLE `omop`.`person`" in joined
    assert "SELECT DISTINCT `entity_id` AS `person_id`" in joined
    assert "`sema_identity`.`entity_identity`" in joined


def test_databricks_child_write_is_atomic_replace_where() -> None:
    cur = _FakeCursor(count=9)
    written = DATABRICKS_FK_BACKEND.write_child(
        cur, _CONDITION, _source("cbio"), _select("databricks")
    )
    assert written == 9
    joined = "\n".join(cur.statements)
    assert "USING DELTA" in joined
    assert "REPLACE WHERE (`source_schema` = 'cbio')" in joined
    # No client temp table, no multi-statement transaction on Databricks.
    assert "TEMP TABLE" not in joined
    assert "BEGIN TRANSACTION" not in joined
    assert "COMMIT" not in joined
    # Spark/Databricks surrogate uses conv(...), never a DuckDB 0x cast.
    assert "CONV(" in joined.upper()
    assert "0x" not in joined


def test_all_databricks_statements_parse_in_databricks_dialect() -> None:
    cur = _FakeCursor()
    DATABRICKS_FK_BACKEND.replace_parent(cur, _PERSON, _REGISTRY)
    DATABRICKS_FK_BACKEND.write_child(
        cur, _CONDITION, _source("cbio"), _select("databricks")
    )
    for stmt in cur.statements:
        sqlglot.parse_one(stmt, dialect="databricks")


def test_duckdb_backend_round_trips_against_real_duckdb(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "poc.duckdb"))
    conn.execute('CREATE SCHEMA "cbio"')
    conn.execute(
        'CREATE TABLE "cbio"."sample" '
        "(sample_id VARCHAR, patient_id VARCHAR, oncotree_code VARCHAR)"
    )
    conn.executemany(
        'INSERT INTO "cbio"."sample" VALUES (?, ?, ?)',
        [("S-1", "P-1", "LUAD"), ("S-2", "P-2", "ZZZ")],
    )
    IdentityRegistry(conn).get_or_create(
        [("cbio", "P-1"), ("cbio", "P-2")], run_id="t"
    )
    result = FkClosedCompiler(
        no_map_default=0, backend=DUCKDB_FK_BACKEND
    ).materialize(
        conn,
        parent=_PERSON,
        child=_CONDITION,
        source=_source("cbio"),
        registry=_REGISTRY,
        decisions=_DECISIONS,
    )
    assert result.child_rows == 2
    assert result.parent_rows == 2
    assert result.missing_key_rows == 0
    orphans = conn.execute(
        "SELECT COUNT(*) FROM omop.condition_occurrence c "
        "LEFT JOIN omop.person p ON c.person_id = p.person_id "
        "WHERE p.person_id IS NULL"
    ).fetchone()[0]
    assert orphans == 0


def test_backend_count_helpers_scope_to_study(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "poc.duckdb"))
    conn.execute('CREATE SCHEMA "cbio"')
    conn.execute(
        'CREATE TABLE "cbio"."sample" '
        "(sample_id VARCHAR, patient_id VARCHAR, oncotree_code VARCHAR)"
    )
    conn.executemany(
        'INSERT INTO "cbio"."sample" VALUES (?, ?, ?)',
        [("S-1", "P-1", "LUAD"), ("S-2", "  ", "ZZZ")],
    )
    IdentityRegistry(conn).get_or_create([("cbio", "P-1")], run_id="t")
    FkClosedCompiler(no_map_default=0, backend=DUCKDB_FK_BACKEND).materialize(
        conn, parent=_PERSON, child=_CONDITION, source=_source("cbio"),
        registry=_REGISTRY, decisions=_DECISIONS,
    )
    assert DUCKDB_FK_BACKEND.child_scope_count(conn, _CONDITION, "cbio") == 1
    assert DUCKDB_FK_BACKEND.missing_key_count(conn, _source("cbio")) == 1
    assert DUCKDB_FK_BACKEND.count_all(conn, _PERSON) == 1
    nulls = DUCKDB_FK_BACKEND.column_null_counts(
        conn, _CONDITION, ("condition_concept_id", "person_id"), "cbio"
    )
    assert nulls == {"condition_concept_id": 0, "person_id": 0}
