"""S1-10: ``sema collapse-omop-identities`` CLI wiring (hermetic DuckDB).

Seeds two MSK studies sharing patient ``P-1``, runs Stage A for each via
``fit-omop-shape``, then the Stage B collapse command, and asserts the duplicate
person is retired and the summary reports it. The live Databricks run is gated on
a second MSK study being ingested; this proves the wiring end-to-end on DuckDB.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest
from click.testing import CliRunner

from sema.cli import cli
from sema.models.planner.lifecycle import Status
from sema.resolve.identity_registry import IdentityRegistry
from sema.resolve.value_mapping_store import ValueMappingStore
from sema.resolve.value_mapping_store_utils import ResolutionStatus, ValueMapping

pytestmark = pytest.mark.unit

_CHORD = "cbioportal_msk_chord_2024"
_IMPACT = "cbioportal_msk_impact_2026"
_POLICY = "omop.oncotree_condition"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _mapping(code: str, concept: int) -> ValueMapping:
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
        resolution_status=ResolutionStatus.RESOLVED,
        no_map_reason=None,
        confidence=1.0,
        status=Status.auto_accepted,
        resolver_policy_ref=_POLICY,
        run_id="r1",
    )


def _seed(path: str) -> None:
    conn = duckdb.connect(path)
    ValueMappingStore(conn).upsert([_mapping("LUAD", 777926), _mapping("GBM", 4000)])
    for schema, rows in (
        (_CHORD, [("C-1", "P-1", "LUAD"), ("C-2", "P-9", "GBM")]),
        (_IMPACT, [("I-1", "P-1", "LUAD"), ("I-2", "P-8", "GBM")]),
    ):
        conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        conn.execute(
            f'CREATE TABLE "{schema}"."sample" '
            "(SAMPLE_ID VARCHAR, PATIENT_ID VARCHAR, ONCOTREE_CODE VARCHAR)"
        )
        conn.executemany(f'INSERT INTO "{schema}"."sample" VALUES (?, ?, ?)', rows)
    conn.close()


def _fit(runner: CliRunner, db: str, schema: str) -> None:
    result = runner.invoke(
        cli,
        ["fit-omop-shape", "--backend", "duckdb", "--duckdb", db, "--study-schema", schema],
    )
    assert result.exit_code == 0, result.output


def _person_count(db: str) -> int:
    conn = duckdb.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM omop_stage_a.person").fetchone()[0]
    conn.close()
    return n


def test_collapse_retires_duplicate_and_reports_summary(
    runner: CliRunner, tmp_path: Path
) -> None:
    db = str(tmp_path / "poc.duckdb")
    _seed(db)
    _fit(runner, db, _CHORD)
    _fit(runner, db, _IMPACT)
    assert _person_count(db) == 4

    result = runner.invoke(
        cli,
        [
            "collapse-omop-identities", "--backend", "duckdb", "--duckdb", db,
            "--study-schema", _CHORD, "--study-schema", _IMPACT,
            "--identity-namespace", "msk_dmp", "--strict",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["collapsed_person_count"] == 1
    assert summary["parent_rows"] == 3
    assert summary["child_rows"] == 4
    assert len(summary["retired_person_ids"]) == 1
    assert _person_count(db) == 3


def test_no_shared_namespace_collapses_nothing(
    runner: CliRunner, tmp_path: Path
) -> None:
    db = str(tmp_path / "poc.duckdb")
    _seed(db)
    _fit(runner, db, _CHORD)
    _fit(runner, db, _IMPACT)

    # Distinct identity namespaces per study -> the P-1 in each is a different
    # person; the command must collapse nothing (over-collapse guard).
    result = runner.invoke(
        cli,
        [
            "collapse-omop-identities", "--backend", "duckdb", "--duckdb", db,
            "--study-schema", _CHORD, "--identity-namespace", _CHORD,
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["collapsed_person_count"] == 0
    assert _person_count(db) == 4


class _FakeDatabricksCursor:
    """Answers the statement shapes the Stage B rebuild issues on Databricks."""

    def __init__(self) -> None:
        self.statements: list[str] = []
        self._rows: list[Any] = []
        self._one: Any = None

    def execute(self, sql: str, parameters: Any = None) -> "_FakeDatabricksCursor":
        self.statements.append(sql)
        self._rows, self._one = [], None
        if "SELECT DISTINCT" in sql:
            self._rows = [("P-1",), ("P-9",)]
        elif "LEFT JOIN" in sql:
            self._one = (0,)  # orphan FK count
        elif "CASE WHEN" in sql:
            self._one = (0, 0, 0)  # required-field null counts
        elif "TRIM(COALESCE" in sql and "COUNT(*)" in sql:
            self._one = (0,)  # missing-key count
        elif "COUNT(*)" in sql:
            self._one = (2,)
        return self

    def fetchall(self) -> list[Any]:
        return self._rows

    def fetchone(self) -> Any:
        return self._one


def test_databricks_backend_collapses_bridges_and_rebuilds(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = str(tmp_path / "store.duckdb")
    _seed(db)
    # Stage B reads the existing registry; seed both studies so a duplicate exists.
    store = duckdb.connect(db)
    IdentityRegistry(store).get_or_create(
        [(_CHORD, "P-1"), (_CHORD, "P-9"), (_IMPACT, "P-1"), (_IMPACT, "P-8")],
        run_id="stage-a",
    )
    store.close()
    cursor = _FakeDatabricksCursor()
    monkeypatch.setattr(
        "showcase.cbioportal_to_omop.cli_omop_collapse.open_databricks_cursor", lambda *a, **k: cursor
    )

    result = runner.invoke(
        cli,
        [
            "collapse-omop-identities", "--backend", "databricks", "--duckdb", db,
            "--study-schema", _CHORD, "--study-schema", _IMPACT,
            "--identity-namespace", "msk_dmp",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["collapsed_person_count"] == 1
    assert summary["registry_rows_bridged"] == 4  # collapse remaps ids, keeps rows
    joined = "\n".join(cursor.statements)
    bridge_idx = next(
        i for i, s in enumerate(cursor.statements)
        if "entity_identity" in s and s.startswith("INSERT")
    )
    parent_idx = next(
        i for i, s in enumerate(cursor.statements)
        if "CREATE OR REPLACE TABLE `omop_stage_a`.`person`" in s
    )
    assert bridge_idx < parent_idx  # collapsed registry bridged before parent read
    assert "REPLACE WHERE" in joined
    assert "BEGIN TRANSACTION" not in joined


def test_missing_duckdb_file_exits_2(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        cli,
        [
            "collapse-omop-identities", "--duckdb", str(tmp_path / "absent.duckdb"),
            "--study-schema", _CHORD, "--identity-namespace", "msk_dmp",
        ],
    )
    assert result.exit_code == 2
    assert "not found" in result.output
