"""S1-08: ``sema fit-omop-shape`` CLI wiring (hermetic).

The DuckDB backend runs the whole FK-closed chain against a seeded in-process
DuckDB (store + registry + source in one file). The Databricks backend uses a
fake cursor (monkeypatched) so the bridge + Delta write path is exercised without
a live warehouse. The live run itself is the end-to-end proof.
"""

from __future__ import annotations

import json
from typing import Any

import duckdb
import pytest
from click.testing import CliRunner

from sema.cli import cli
from sema.models.planner.lifecycle import Status
from sema.resolve.value_mapping_store import ValueMappingStore
from sema.resolve.value_mapping_store_utils import ResolutionStatus, ValueMapping

pytestmark = pytest.mark.unit

_SCHEMA = "cbioportal_msk_chord_2024"
_POLICY = "omop.oncotree_condition"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _mapping(code: str, concept: int | None) -> ValueMapping:
    no_map = concept is None
    return ValueMapping(
        source_vocabulary="OncoTree",
        normalized_source_value=code,
        target_property_ref="target.condition_occurrence_staging.condition_concept_id",
        target_field="condition_concept_id",
        vocab_binding="OMOP-Condition",
        concept_id=concept,
        vocab_release="omop-vocab-2024",
        valid_start=None,
        valid_end=None,
        resolution_status=ResolutionStatus.NO_MAP if no_map else ResolutionStatus.RESOLVED,
        no_map_reason="no standard candidate" if no_map else None,
        confidence=1.0,
        status=Status.auto_accepted,
        resolver_policy_ref=_POLICY,
        run_id="r1",
    )


def _seed_store(path: str) -> None:
    conn = duckdb.connect(path)
    ValueMappingStore(conn).upsert(
        [_mapping("LUAD", 777926), _mapping("GBM", 4000), _mapping("ZZZ", None)]
    )
    conn.close()


def _seed_source(path: str) -> None:
    conn = duckdb.connect(path)
    conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{_SCHEMA}"')
    conn.execute(
        f'CREATE TABLE "{_SCHEMA}"."sample" '
        "(SAMPLE_ID VARCHAR, PATIENT_ID VARCHAR, ONCOTREE_CODE VARCHAR)"
    )
    conn.executemany(
        f'INSERT INTO "{_SCHEMA}"."sample" VALUES (?, ?, ?)',
        [("S-1", "P-1", "LUAD"), ("S-2", "P-2", "GBM"), ("S-3", "P-1", "ZZZ")],
    )
    conn.close()


def test_is_registered(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["fit-omop-shape", "--help"])
    assert result.exit_code == 0
    assert "--study-schema" in result.output
    assert "--omop-schema" in result.output


def test_missing_duckdb_exits_2(runner: CliRunner, tmp_path) -> None:
    result = runner.invoke(
        cli,
        ["fit-omop-shape", "--duckdb", str(tmp_path / "nope.duckdb"),
         "--study-schema", _SCHEMA],
    )
    assert result.exit_code == 2
    assert "not found" in result.output


def test_no_decisions_exits_2(runner: CliRunner, tmp_path) -> None:
    db = str(tmp_path / "poc.duckdb")
    _seed_source(db)  # source but no value-mapping decisions
    result = runner.invoke(
        cli, ["fit-omop-shape", "--duckdb", db, "--study-schema", _SCHEMA]
    )
    assert result.exit_code == 2
    assert "no value-mapping decisions" in result.output


def test_duckdb_backend_runs_full_chain(runner: CliRunner, tmp_path) -> None:
    db = str(tmp_path / "poc.duckdb")
    _seed_store(db)
    _seed_source(db)
    result = runner.invoke(
        cli,
        ["fit-omop-shape", "--duckdb", db, "--study-schema", _SCHEMA, "--strict"],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["parent_rows"] == 2
    assert summary["child_rows"] == 3
    assert summary["missing_key_rows"] == 0
    assert summary["gate_d_lite"]["outcome"] == "PASS"
    conn = duckdb.connect(db)
    concepts = {
        r[0]
        for r in conn.execute(
            "SELECT condition_concept_id FROM omop_stage_a.condition_occurrence"
        ).fetchall()
    }
    conn.close()
    assert concepts == {777926, 4000, 0}  # ZZZ NO_MAP -> 0 (D8)


class _FakeDatabricksCursor:
    """Answers the statement shapes the live chain issues (enum, counts, write)."""

    def __init__(self) -> None:
        self.statements: list[str] = []
        self._rows: list[Any] = []
        self._one: Any = None

    def execute(self, sql: str, parameters: Any = None) -> "_FakeDatabricksCursor":
        self.statements.append(sql)
        self._rows, self._one = [], None
        if "SELECT DISTINCT" in sql:
            self._rows = [("P-1",), ("P-2",)]
        elif "LEFT JOIN" in sql:
            self._one = (0,)  # orphan FK count
        elif "CASE WHEN" in sql:
            self._one = (0, 0, 0)  # required-field null counts
        elif "TRIM(COALESCE" in sql and "COUNT(*)" in sql:
            self._one = (0,)  # missing-key count
        elif "COUNT(*)" in sql:
            self._one = (3,)  # source total / parent / child-scope counts
        return self

    def fetchall(self) -> list[Any]:
        return self._rows

    def fetchone(self) -> Any:
        return self._one


def test_databricks_backend_bridges_and_writes(
    runner: CliRunner, tmp_path, monkeypatch
) -> None:
    db = str(tmp_path / "store.duckdb")
    _seed_store(db)
    cursor = _FakeDatabricksCursor()
    monkeypatch.setattr(
        "showcase.cbioportal_to_omop.cli_omop_shape.open_databricks_cursor", lambda *a, **k: cursor
    )
    result = runner.invoke(
        cli,
        ["fit-omop-shape", "--backend", "databricks", "--duckdb", db,
         "--study-schema", _SCHEMA, "--omop-schema", "omop_stage_a"],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["registry_rows_bridged"] == 2
    assert summary["gate_d_lite"]["outcome"] == "PASS"
    joined = "\n".join(cursor.statements)
    # Registry bridged into Delta BEFORE the parent rebuild reads it.
    bridge_idx = next(
        i for i, s in enumerate(cursor.statements)
        if "entity_identity" in s and s.startswith("INSERT")
    )
    parent_idx = next(
        i for i, s in enumerate(cursor.statements)
        if "CREATE OR REPLACE TABLE `omop_stage_a`.`person`" in s
    )
    assert bridge_idx < parent_idx
    assert "REPLACE WHERE" in joined
    assert "USING DELTA" in joined
    assert "BEGIN TRANSACTION" not in joined
