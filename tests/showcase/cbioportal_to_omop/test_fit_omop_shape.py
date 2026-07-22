"""S1-08 — the FK-closed OMOP-shape chain (DuckDB end-to-end + Databricks wiring).

The DuckDB path runs against a real in-process DuckDB and proves the full row-count
identity + FK closure + idempotency the live run asserts. The Databricks path uses
a fake cursor (registry rows come from a real DuckDB registry) to prove the bridge
runs BEFORE the parent/child write and that every emitted statement is Delta SQL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import duckdb
import pytest

from sema.compile.compiler_utils import StagingDecision
from sema.compile.fk_backend import DATABRICKS_FK_BACKEND, DUCKDB_FK_BACKEND
from sema.compile.fk_closed_compiler_utils import ChildSourceSpec, RegistryJoinSpec
from sema.pipeline.fk_closed_fit import FkClosedFitRequest, run_fk_closed_fit
from sema.resolve.identity_registry import (
    DEFAULT_SCHEMA,
    DEFAULT_TABLE,
    IdentityRegistry,
)
from showcase.cbioportal_to_omop.omop_policy import MISSING_PERSON_KEY_REASON, make_omop_fk_specs

pytestmark = pytest.mark.unit

_PARENT, _CHILD, _REQUIRED = make_omop_fk_specs("omop_stage_a")
_REGISTRY_SPEC = RegistryJoinSpec(
    schema=DEFAULT_SCHEMA,
    table=DEFAULT_TABLE,
    namespace_column="source_namespace",
    key_column="source_entity_key",
    id_column="entity_id",
)
_DECISIONS = [
    StagingDecision("LUAD", 777926, "RESOLVED", None, "auto_accepted"),
    StagingDecision("GBM", 4000, "RESOLVED", None, "auto_accepted"),
    StagingDecision("ZZZ", None, "NO_MAP", "no standard candidate", "auto_accepted"),
]
_SCHEMA = "cbioportal_msk_chord_2024"


def _source() -> ChildSourceSpec:
    return ChildSourceSpec(
        schema=_SCHEMA,
        table="sample",
        value_column="oncotree_code",
        row_ref_column="sample_id",
        patient_key_column="patient_id",
    )


def _request(keys: list[str], row_count: int) -> FkClosedFitRequest:
    return FkClosedFitRequest(
        source=_source(),
        source_row_count=row_count,
        distinct_patient_keys=keys,
        parent=_PARENT,
        child=_CHILD,
        registry_spec=_REGISTRY_SPEC,
        decisions=_DECISIONS,
        required_fields=_REQUIRED,
        no_map_default=0,
        missing_key_reason=MISSING_PERSON_KEY_REASON,
        run_id="s1-08",
    )


def _seed_source(conn: duckdb.DuckDBPyConnection, rows: list[tuple[str, str | None, str]]) -> None:
    conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{_SCHEMA}"')
    conn.execute(
        f'CREATE OR REPLACE TABLE "{_SCHEMA}"."sample" '
        "(sample_id VARCHAR, patient_id VARCHAR, oncotree_code VARCHAR)"
    )
    conn.executemany(f'INSERT INTO "{_SCHEMA}"."sample" VALUES (?, ?, ?)', rows)


class _FakeCursor:
    """Records SQL; every scalar read returns 0 so the FK-closure assert passes."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, sql: str, parameters: Sequence[Any] | None = None) -> "_FakeCursor":
        self.statements.append(sql)
        return self

    def fetchone(self) -> tuple[int, ...]:
        return (0, 0, 0, 0)  # wide enough for the multi-column null-count read


def test_duckdb_end_to_end_row_count_identity_and_fk_closure(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "poc.duckdb"))
    _seed_source(
        conn,
        [("S-1", "P-1", "LUAD"), ("S-2", "P-2", "GBM"), ("S-3", "P-1", "ZZZ")],
    )
    registry = IdentityRegistry(conn)
    result = run_fk_closed_fit(
        _request(["P-1", "P-2"], row_count=3),
        registry=registry,
        target_cursor=conn,
        backend=DUCKDB_FK_BACKEND,
        bridge=False,
    )
    assert result.fk.child_rows == 3
    assert result.fk.parent_rows == 2
    assert result.fk.missing_key_rows == 0
    assert result.registry_rows_bridged == 0
    assert result.qa.passed  # closure + required-not-null + row-count identity
    # The ZZZ (NO_MAP) row lands with concept_id 0 (D8), FK-closed.
    concepts = {
        r[0]
        for r in conn.execute(
            "SELECT condition_concept_id FROM omop_stage_a.condition_occurrence"
        ).fetchall()
    }
    assert 0 in concepts
    orphans = conn.execute(
        "SELECT COUNT(*) FROM omop_stage_a.condition_occurrence c "
        "LEFT JOIN omop_stage_a.person p ON c.person_id = p.person_id "
        "WHERE p.person_id IS NULL"
    ).fetchone()[0]
    assert orphans == 0


def test_duckdb_missing_key_accounts_in_row_count_identity(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "poc.duckdb"))
    _seed_source(
        conn, [("S-1", "P-1", "LUAD"), ("S-2", "  ", "GBM"), ("S-3", None, "GBM")]
    )
    result = run_fk_closed_fit(
        _request(["P-1"], row_count=3),
        registry=IdentityRegistry(conn),
        target_cursor=conn,
        backend=DUCKDB_FK_BACKEND,
    )
    # written(1) + missing(2) == source(3); still passes Gate-D-lite.
    assert result.fk.child_rows == 1
    assert result.fk.missing_key_rows == 2
    assert result.qa.passed


def test_duckdb_rerun_is_idempotent(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "poc.duckdb"))
    _seed_source(conn, [("S-1", "P-1", "LUAD"), ("S-2", "P-2", "GBM")])
    req = _request(["P-1", "P-2"], row_count=2)
    first = run_fk_closed_fit(req, registry=IdentityRegistry(conn), target_cursor=conn)
    second = run_fk_closed_fit(req, registry=IdentityRegistry(conn), target_cursor=conn)
    assert first.fk.child_rows == second.fk.child_rows == 2
    total = conn.execute(
        "SELECT COUNT(*) FROM omop_stage_a.condition_occurrence"
    ).fetchone()[0]
    assert total == 2  # scoped swap replaced, did not duplicate


def test_databricks_bridges_registry_before_write(tmp_path: Path) -> None:
    registry = IdentityRegistry(duckdb.connect(str(tmp_path / "poc.duckdb")))
    registry.get_or_create([(_SCHEMA, "P-1"), (_SCHEMA, "P-2")], run_id="seed")
    cur = _FakeCursor()
    result = run_fk_closed_fit(
        _request(["P-1", "P-2"], row_count=5),
        registry=registry,
        target_cursor=cur,
        backend=DATABRICKS_FK_BACKEND,
        bridge=True,
    )
    assert result.registry_rows_bridged == 2
    joined = "\n".join(cur.statements)
    # Bridge (registry mirror) happens before the parent rebuild reads it.
    bridge_idx = next(
        i for i, s in enumerate(cur.statements) if "entity_identity" in s and "INSERT" in s
    )
    parent_idx = next(
        i for i, s in enumerate(cur.statements)
        if "CREATE OR REPLACE TABLE `omop_stage_a`.`person`" in s
    )
    assert bridge_idx < parent_idx
    assert "REPLACE WHERE" in joined  # Delta child scoped-swap
    assert "BEGIN TRANSACTION" not in joined
