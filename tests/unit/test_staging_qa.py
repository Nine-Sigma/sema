"""US-011: Gate D-lite staging QA — hermetic unit tests (TDD-first).

Covers the three §1.5 staging checks (row count, null-rate reconciled against
``resolution_status``, NO_MAP accounting reconciled against the US-002 gold set)
as pure functions, plus the DuckDB reader/orchestration over a real temp file
(in-process DuckDB is local, so it stays within the hermetic-unit rule).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from sema.compile.compiler_utils import create_staging_table_sql, staging_column_order
from sema.eval.mapping_goldset import GoldSet
from sema.eval.mapping_goldset_utils import GoldLabel, GoldRow
from sema.eval.staging_qa import read_staging_rows, run_staging_qa, staging_scope_count
from sema.eval.staging_qa_utils import (
    QAOutcome,
    StagingQAReport,
    StagingRow,
    check_no_map_accounting,
    check_null_rate,
    check_row_count,
)
from sema.resolve.policies.omop import OMOP_STAGING_COLUMNS

pytestmark = pytest.mark.unit


_COLS = OMOP_STAGING_COLUMNS


def _gold(*rows: tuple[str, int | None, GoldLabel]) -> GoldSet:
    return GoldSet(
        rows=[
            GoldRow(oncotree_code=c, gold_concept_id=cid, gold_label=lbl, row_count=1)
            for c, cid, lbl in rows
        ]
    )


# --- check_row_count --------------------------------------------------------


def test_row_count_match_passes() -> None:
    check = check_row_count(actual=100, expected=100)
    assert check.outcome is QAOutcome.PASS
    assert check.passed


def test_row_count_mismatch_fails_with_reason() -> None:
    check = check_row_count(actual=97, expected=100)
    assert check.outcome is QAOutcome.FAIL
    assert check.reason is not None
    assert check.details["actual"] == 97 and check.details["expected"] == 100


# --- check_null_rate (reconciled against resolution_status) -----------------


def test_null_rate_passes_when_nulls_match_no_map() -> None:
    rows = [
        StagingRow("LUAD", 4314337, "RESOLVED"),
        StagingRow("MIXED", None, "NO_MAP"),
    ]
    check = check_null_rate(rows)
    assert check.outcome is QAOutcome.PASS
    assert check.details["null_rows"] == 1 and check.details["no_map_rows"] == 1


def test_null_rate_fails_on_resolved_but_null() -> None:
    rows = [
        StagingRow("LUAD", None, "RESOLVED"),
        StagingRow("MIXED", None, "NO_MAP"),
    ]
    check = check_null_rate(rows)
    assert check.outcome is QAOutcome.FAIL
    assert check.details["resolved_but_null"] == ["LUAD"]


def test_null_rate_fails_on_no_map_but_populated() -> None:
    rows = [StagingRow("MIXED", 999, "NO_MAP")]
    check = check_null_rate(rows)
    assert check.outcome is QAOutcome.FAIL
    assert check.details["no_map_but_populated"] == ["MIXED"]


# --- check_no_map_accounting (reconciled against gold set) ------------------


def test_no_map_accounting_passes_when_no_map_matches_gold() -> None:
    rows = [
        StagingRow("LUAD", 4314337, "RESOLVED"),
        StagingRow("MIXED", None, "NO_MAP"),
    ]
    gold = _gold(
        ("LUAD", 4314337, GoldLabel.RESOLVED),
        ("MIXED", None, GoldLabel.NO_MAP),
    )
    check = check_no_map_accounting(rows, gold)
    assert check.outcome is QAOutcome.PASS


def test_no_map_accounting_fails_when_gold_resolved_but_staged_no_map() -> None:
    rows = [StagingRow("LUAD", None, "NO_MAP")]
    gold = _gold(("LUAD", 4314337, GoldLabel.RESOLVED))
    check = check_no_map_accounting(rows, gold)
    assert check.outcome is QAOutcome.FAIL
    assert check.details["gold_resolved_but_no_map"] == ["LUAD"]


def test_no_map_accounting_skips_unlabelled_gold() -> None:
    rows = [StagingRow("XYZ", None, "NO_MAP")]
    gold = _gold(("XYZ", None, GoldLabel.UNLABELLED))
    check = check_no_map_accounting(rows, gold)
    assert check.outcome is QAOutcome.PASS


# --- report aggregation -----------------------------------------------------


def test_report_passed_only_when_all_checks_pass() -> None:
    ok = StagingQAReport(
        checks=(check_row_count(1, 1), check_null_rate([StagingRow("A", 1, "RESOLVED")])),
    )
    assert ok.passed and ok.outcome is QAOutcome.PASS
    bad = StagingQAReport(checks=(check_row_count(1, 2),))
    assert not bad.passed
    assert [c.name for c in bad.failures()] == ["row_count"]
    assert bad.as_dict()["outcome"] == "FAIL"


# --- DuckDB reader + orchestration over a temp staging table ----------------


def _write_staging(conn: duckdb.DuckDBPyConnection, schema: str, table: str) -> None:
    conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    conn.execute(create_staging_table_sql(_COLS, schema, table))
    order = staging_column_order(_COLS)
    placeholders = ", ".join("?" for _ in order)
    cols = ", ".join(f'"{c}"' for c in order)
    insert = f'INSERT INTO "{schema}"."{table}" ({cols}) VALUES ({placeholders})'
    base = {
        _COLS.source_schema: "study_a",
        _COLS.source_table: "sample",
        _COLS.source_row_ref: None,
        _COLS.source_patient_key: None,
        _COLS.resolver_policy_ref: "p",
        _COLS.vocab_release: "v",
        _COLS.no_map_reason: None,
        _COLS.status_column: "auto_accepted",
        _COLS.run_id: "r",
    }
    rows = [
        {**base, _COLS.source_value_column: "LUAD", _COLS.target_concept_column: 4314337,
         _COLS.resolution_status: "RESOLVED"},
        {**base, _COLS.source_value_column: "MIXED", _COLS.target_concept_column: None,
         _COLS.no_map_reason: "no standard target", _COLS.resolution_status: "NO_MAP"},
    ]
    for row in rows:
        conn.execute(insert, [row[c] for c in order])


def test_read_staging_rows_and_run_qa(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "qa.duckdb"))
    _write_staging(conn, "sema_staging", "condition_staging")

    rows = read_staging_rows(
        conn, _COLS, "sema_staging", "condition_staging", "study_a", "sample"
    )
    assert {r.source_value for r in rows} == {"LUAD", "MIXED"}
    assert staging_scope_count(
        conn, _COLS, "sema_staging", "condition_staging", "study_a", "sample"
    ) == 2

    gold = _gold(
        ("LUAD", 4314337, GoldLabel.RESOLVED),
        ("MIXED", None, GoldLabel.NO_MAP),
    )
    report = run_staging_qa(
        conn,
        columns=_COLS,
        staging_schema="sema_staging",
        staging_table="condition_staging",
        source_schema="study_a",
        source_table="sample",
        expected_row_count=2,
        gold_set=gold,
    )
    assert report.passed
    assert {c.name for c in report.checks} == {
        "row_count",
        "null_rate",
        "no_map_accounting",
    }


def test_run_qa_fails_on_corrupted_row_count(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "qa.duckdb"))
    _write_staging(conn, "sema_staging", "condition_staging")
    report = run_staging_qa(
        conn,
        columns=_COLS,
        staging_schema="sema_staging",
        staging_table="condition_staging",
        source_schema="study_a",
        source_table="sample",
        expected_row_count=999,
        gold_set=_gold(("LUAD", 4314337, GoldLabel.RESOLVED)),
    )
    assert not report.passed
    assert "row_count" in {c.name for c in report.failures()}
