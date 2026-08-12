"""CLI surface for the gold-set snapshot lifecycle (G-03/G-05/G-06).

The three refresh operations are separate commands rather than flags on one, so
the command a person types records which blast radius they intended.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb
import pytest
from click.testing import CliRunner

from sema.cli_eval import eval_group
from sema.eval.goldset_snapshot import load_snapshot, resolve_current_version, snapshot_dir
from sema.eval.goldset_snapshot_utils import GoldSetHeader, UniverseEntry
from sema.eval.goldset_snapshot import write_snapshot
from sema.eval.goldset_source import SourceKind, SourceSpec
from sema.eval.goldset_worksheet import WORKSHEET_COLUMNS
from sema.eval.mapping_goldset_utils import GoldLabel, GoldRow, TierState

pytestmark = pytest.mark.unit


_SPEC = SourceSpec(
    kind=SourceKind.STAGING,
    table="sema_staging.condition_staging",
    code_column="source_oncotree_code",
    scope_column="source_schema",
    scope_values=("study_a",),
)
_COUNTS = {"LUAD": 100, "COAD": 60, "RARE": 4, "ODD": 1}


@pytest.fixture()
def gold_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "gold"
    header = GoldSetHeader(
        snapshot_version="v1",
        snapshot_date="2026-08-11",
        source_of_truth=_SPEC,
        target_vocabulary="SNOMED",
        target_domain="Condition",
        vocab_release="omop-vocab-2024",
        oracle_source="human curation",
        oracle_version="unlabelled",
        tier_codes=("LUAD", "COAD"),
        challenge_codes=("ODD",),
        tier_target_row_share=0.95,
        tier_achieved_row_share=160 / 165,
        min_frozen_tier_row_share=0.90,
        max_unseen_code_share=0.10,
    )
    rows = [
        GoldRow("LUAD", None, GoldLabel.UNLABELLED, 100, tier_state=TierState.IN_TIER),
        GoldRow("COAD", None, GoldLabel.UNLABELLED, 60, tier_state=TierState.IN_TIER),
        GoldRow("ODD", None, GoldLabel.UNLABELLED, 1, tier_state=TierState.CHALLENGE),
        GoldRow("RARE", None, GoldLabel.UNLABELLED, 4, tier_state=TierState.OUT_OF_TIER),
    ]
    universe = tuple(UniverseEntry(c, n) for c, n in _COUNTS.items())
    write_snapshot(snapshot_dir("v1", root), header, rows, universe)
    (root / "current.json").write_text('{"snapshot_version": "v1"}\n', encoding="utf-8")
    monkeypatch.setattr("sema.cli_goldset.GOLD_ROOT", root)
    return root


@pytest.fixture()
def db(tmp_path: Path) -> str:
    path = tmp_path / "live.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA sema_staging")
    con.execute(
        "CREATE TABLE sema_staging.condition_staging "
        "(source_schema VARCHAR, source_oncotree_code VARCHAR)"
    )
    con.execute("CREATE SCHEMA vocabulary_omop")
    con.execute(
        "CREATE TABLE vocabulary_omop.concept "
        "(concept_code VARCHAR, concept_name VARCHAR, vocabulary_id VARCHAR)"
    )
    con.execute(
        "INSERT INTO vocabulary_omop.concept VALUES ('LUAD', 'Lung Adenocarcinoma', 'OncoTree')"
    )
    con.execute("CREATE TABLE _counts (code VARCHAR, n BIGINT)")
    con.executemany("INSERT INTO _counts VALUES (?, ?)", [[c, n] for c, n in _COUNTS.items()])
    con.execute(
        "INSERT INTO sema_staging.condition_staging "
        "SELECT 'study_a', c.code FROM _counts c, range(c.n)"
    )
    con.close()
    return str(path)


def _run(*args: str) -> str:
    result = CliRunner().invoke(eval_group, list(args))
    assert result.exit_code == 0, result.output + str(result.exception)
    return result.output


def test_drift_reports_a_fresh_benchmark_as_json(gold_root: Path, db: str) -> None:
    payload = json.loads(_run("goldset", "drift", "--db", db))

    assert payload["scope_changed"] is False
    assert payload["drift"]["full_universe"] == 0.0
    assert payload["freshness"]["is_stale"] is False


def test_drift_names_a_stale_benchmark_as_stale_not_drifted(gold_root: Path, db: str) -> None:
    con = duckdb.connect(db)
    con.execute(
        "INSERT INTO sema_staging.condition_staging "
        "SELECT 'study_a', 'FLOOD' FROM range(9000)"
    )
    con.close()

    payload = json.loads(_run("goldset", "drift", "--db", db))

    assert payload["freshness"]["is_stale"] is True
    assert "benchmark stale" in payload["freshness"]["reason"]
    assert "re-tier" in payload["freshness"]["reason"]


def test_observe_publishes_a_new_version_and_advances_the_pointer(
    gold_root: Path, db: str
) -> None:
    before = (snapshot_dir("v1", gold_root) / "oncotree_condition_slice0.jsonl").read_bytes()

    output = _run("goldset", "observe", "--db", db, "--version", "v2", "--date", "2026-09-01")

    assert "published v2" in output
    assert "labels are a human gate" in output
    assert resolve_current_version(gold_root) == "v2"
    assert (snapshot_dir("v1", gold_root) / "oncotree_condition_slice0.jsonl").read_bytes() == before


def test_re_tier_moves_the_frozen_head(gold_root: Path, db: str) -> None:
    _run(
        "goldset", "re-tier", "--db", db, "--version", "v2", "--date", "2026-09-01",
        "--target-row-share", "0.6",
    )

    assert load_snapshot(snapshot_dir("v2", gold_root)).header.tier_codes == ("LUAD",)


def test_re_scope_retires_codes_that_leave_the_declared_scope(
    gold_root: Path, db: str
) -> None:
    con = duckdb.connect(db)
    con.execute("UPDATE sema_staging.condition_staging SET source_schema = 'study_b'")
    con.execute("DELETE FROM sema_staging.condition_staging WHERE source_oncotree_code = 'LUAD'")
    con.close()

    _run(
        "goldset", "re-scope", "--db", db, "--version", "v2", "--date", "2026-09-01",
        "--kind", "staging", "--table", "sema_staging.condition_staging",
        "--code-column", "source_oncotree_code", "--scope-column", "source_schema",
        "--scope-value", "study_b",
    )
    snapshot = load_snapshot(snapshot_dir("v2", gold_root))

    assert snapshot.by_code()["LUAD"].tier_state is TierState.RETIRED


def test_the_worksheet_is_blank_and_carries_source_context(
    gold_root: Path, db: str, tmp_path: Path
) -> None:
    output_path = tmp_path / "ws.csv"

    output = _run(
        "goldset", "worksheet", "--db", db, "--head-size", "2", "--output", str(output_path)
    )
    rows = list(csv.DictReader(output_path.open(encoding="utf-8")))

    assert "no candidate target concepts are pre-filled" in output
    assert {r["oncotree_code"] for r in rows} == {"LUAD", "COAD", "ODD"}
    assert {r["oncotree_name"] for r in rows} == {"Lung Adenocarcinoma", ""}
    assert all(r["gold_concept_id"] == "" for r in rows)


def test_apply_labels_publishes_a_labelled_snapshot(
    gold_root: Path, db: str, tmp_path: Path
) -> None:
    worksheet = tmp_path / "done.csv"
    with worksheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(WORKSHEET_COLUMNS))
        writer.writeheader()
        writer.writerow(
            {
                "oncotree_code": "ODD",
                "gold_label": "NO_MAP",
                "curator": "dean",
                "review_date": "2026-09-01",
                "evidence": "no acceptable Condition target in SNOMED@omop-vocab-2024",
                "notes": "",
            }
        )

    output = _run(
        "goldset", "apply-labels", "--worksheet", str(worksheet),
        "--version", "v2", "--date", "2026-09-01",
    )
    snapshot = load_snapshot(snapshot_dir("v2", gold_root))

    assert snapshot.by_code()["ODD"].gold_label is GoldLabel.NO_MAP
    assert "lack a second reviewer" in output, "the unadjudicated gap is surfaced"


def test_apply_labels_refuses_a_label_without_provenance(
    gold_root: Path, tmp_path: Path
) -> None:
    worksheet = tmp_path / "bad.csv"
    with worksheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(WORKSHEET_COLUMNS))
        writer.writeheader()
        writer.writerow({"oncotree_code": "LUAD", "gold_label": "NO_MAP"})

    result = CliRunner().invoke(
        eval_group,
        ["goldset", "apply-labels", "--worksheet", str(worksheet),
         "--version", "v2", "--date", "2026-09-01"],
    )

    assert result.exit_code != 0
    assert "annotation floor" in str(result.exception)
