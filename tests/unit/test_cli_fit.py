"""US-012A: ``sema fit`` CLI registration + wiring (hermetic).

The full chain is covered by ``test_fit_slice0`` (run_fit) and the live
integration test; here we assert the command is registered, exposes the study /
manifest options, and fails cleanly when the DuckDB file is missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest
from click.testing import CliRunner

from sema.cli import cli

pytestmark = pytest.mark.unit

_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "sema"
    / "targets"
    / "manifests"
    / "omop_condition_slice0.yaml"
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _seed_fixture_db(path: Path) -> None:
    """A tiny all-VARCHAR OMOP vocabulary + source study (mirrors poc.duckdb)."""
    conn = duckdb.connect(str(path))
    conn.execute('CREATE SCHEMA "vocabulary_omop"')
    conn.execute(
        'CREATE TABLE "vocabulary_omop"."concept" '
        "(concept_id VARCHAR, concept_name VARCHAR, domain_id VARCHAR, "
        "vocabulary_id VARCHAR, standard_concept VARCHAR, concept_code VARCHAR, "
        "invalid_reason VARCHAR)"
    )
    conn.executemany(
        'INSERT INTO "vocabulary_omop"."concept" VALUES (?, ?, ?, ?, ?, ?, ?)',
        [
            ("777926", "Lung Adenocarcinoma", "Condition", "OncoTree", None, "LUAD", None),
            ("45768916", "Adenocarcinoma of lung", "Condition", "SNOMED", "S", "254626006", None),
        ],
    )
    conn.execute(
        'CREATE TABLE "vocabulary_omop"."concept_relationship" '
        "(concept_id_1 VARCHAR, concept_id_2 VARCHAR, relationship_id VARCHAR)"
    )
    conn.execute(
        'INSERT INTO "vocabulary_omop"."concept_relationship" VALUES '
        "('777926', '45768916', 'Maps to')"
    )
    conn.execute('CREATE SCHEMA "study"')
    conn.execute('CREATE TABLE "study"."sample" (ONCOTREE_CODE VARCHAR)')
    conn.executemany(
        'INSERT INTO "study"."sample" VALUES (?)',
        [("LUAD",), ("LUAD",), ("ZZZZ",)],
    )
    conn.close()


def test_fit_runs_full_chain_on_duckdb(runner: CliRunner, tmp_path) -> None:
    db = tmp_path / "fixture.duckdb"
    _seed_fixture_db(db)
    result = runner.invoke(
        cli,
        [
            "fit",
            "--manifest",
            str(_MANIFEST),
            "--duckdb",
            str(db),
            "--study-schema",
            "study",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["rows_staged"] == 3
    assert summary["gate_d_lite"]["outcome"] == "PASS"
    assert summary["staging"] == "sema_staging.condition_staging"
    assert "eval" in summary
    # the staging table was actually written into the DuckDB file
    conn = duckdb.connect(str(db))
    total = conn.execute(
        'SELECT COUNT(*) FROM "sema_staging"."condition_staging"'
    ).fetchone()[0]
    conn.close()
    assert total == 3


def test_fit_strict_passes_on_conformance_without_gold(
    runner: CliRunner, tmp_path
) -> None:
    db = tmp_path / "fixture.duckdb"
    _seed_fixture_db(db)
    result = runner.invoke(
        cli,
        [
            "fit",
            "--manifest",
            str(_MANIFEST),
            "--duckdb",
            str(db),
            "--study-schema",
            "study",
            "--strict",
        ],
    )
    # No gold labels, but every resolved concept is valid + standard + Condition,
    # so contract conformance passes -> strict passes. Gold no longer gates.
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["conformance"]["passed"] is True
    assert summary["conformance"]["violations"] == []


def test_fit_strict_fails_on_labelled_gold_contradiction(
    runner: CliRunner, tmp_path
) -> None:
    db = tmp_path / "fixture.duckdb"
    _seed_fixture_db(db)
    gold = tmp_path / "gold.jsonl"
    # LUAD resolves to 45768916, but the human label says 999 -> `wrong` cell.
    gold.write_text(
        json.dumps(
            {
                "oncotree_code": "LUAD",
                "gold_concept_id": 999,
                "gold_label": "RESOLVED",
                "row_count": 1,
                "notes": "deliberately wrong",
            }
        )
        + "\n"
    )
    result = runner.invoke(
        cli,
        [
            "fit",
            "--manifest",
            str(_MANIFEST),
            "--duckdb",
            str(db),
            "--study-schema",
            "study",
            "--gold",
            str(gold),
            "--strict",
        ],
    )
    assert result.exit_code == 3, result.output
    assert "labelled gold contradiction" in result.output


def test_fit_is_registered(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["fit", "--help"])
    assert result.exit_code == 0
    assert "--manifest" in result.output
    assert "--study-schema" in result.output
    assert "--value-column" in result.output


def test_fit_missing_duckdb_exits_2(runner: CliRunner, tmp_path) -> None:
    manifest = tmp_path / "m.yaml"
    manifest.write_text("manifest_version: 1\n")
    result = runner.invoke(
        cli,
        [
            "fit",
            "--manifest",
            str(manifest),
            "--duckdb",
            str(tmp_path / "does_not_exist.duckdb"),
        ],
    )
    assert result.exit_code == 2
    assert "not found" in result.output


# --- US-013: Databricks backend (hermetic, fake cursor) --------------------


class _FakeDatabricksCursor:
    """A faithful stand-in for a Databricks SQL cursor.

    Answers the handful of statement shapes the chain issues (source
    enumeration, vocabulary lookups, staging write + QA read) so the whole
    databricks path is exercised without a live warehouse.
    """

    def __init__(self) -> None:
        self.statements: list[str] = []
        self._rows: list[Any] = []
        self._one: Any = None

    def execute(self, sql: str, parameters: Any = None) -> "_FakeDatabricksCursor":
        self.statements.append(sql)
        params = list(parameters) if parameters else []
        self._rows, self._one = [], None
        if "SELECT DISTINCT" in sql:
            self._rows = [("LUAD",), ("ZZZZ",)]
        elif "COUNT(*)" in sql:
            self._one = (3,)
        elif len(params) == 3:  # maps_to_targets(concept_id, rel, standard_flag)
            self._rows = [
                ("45768916", "Adeno of lung", "Condition", "SNOMED", "S", "254", None)
            ]
        elif len(params) == 2:  # concept_by_code(vocabulary, code)
            if params[1] == "LUAD":
                self._rows = [
                    ("777926", "Lung Adeno", "Condition", "OncoTree", None, "LUAD", None)
                ]
        elif "`sema_staging`.`condition_staging`" in sql:  # QA read-back
            self._rows = [
                ("LUAD", 45768916, "RESOLVED"),
                ("LUAD", 45768916, "RESOLVED"),
                ("ZZZZ", None, "NO_MAP"),
            ]
        return self

    def fetchall(self) -> list[Any]:
        return self._rows

    def fetchone(self) -> Any:
        return self._one


def test_fit_databricks_backend_runs_chain(runner: CliRunner, tmp_path, monkeypatch) -> None:
    cursor = _FakeDatabricksCursor()
    monkeypatch.setattr(
        "sema.cli_fit.open_databricks_cursor", lambda *a, **k: cursor
    )
    result = runner.invoke(
        cli,
        [
            "fit",
            "--manifest",
            str(_MANIFEST),
            "--backend",
            "databricks",
            "--study-schema",
            "study",
            "--duckdb",
            str(tmp_path / "store.duckdb"),
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["rows_staged"] == 3
    assert summary["gate_d_lite"]["outcome"] == "PASS"
    assert summary["staging"] == "sema_staging.condition_staging"
    # the staging write went through the atomic Delta REPLACE WHERE path
    joined = "\n".join(cursor.statements)
    assert "REPLACE WHERE" in joined
    assert "USING DELTA" in joined


def test_fit_databricks_requires_study_schema(runner: CliRunner, tmp_path) -> None:
    result = runner.invoke(
        cli,
        [
            "fit",
            "--manifest",
            str(_MANIFEST),
            "--backend",
            "databricks",
            "--duckdb",
            str(tmp_path / "store.duckdb"),
        ],
    )
    assert result.exit_code == 2
    assert "study-schema" in result.output
