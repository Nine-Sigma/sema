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
from sema.models.planner.lifecycle import Status

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
        "(concept_id BIGINT, concept_code VARCHAR, concept_name VARCHAR, "
        "vocabulary_id VARCHAR, domain_id VARCHAR, standard_concept VARCHAR)"
    )
    con.executemany(
        "INSERT INTO vocabulary_omop.concept VALUES (?, ?, ?, ?, ?, ?)",
        [
            [777926, "LUAD", "Lung Adenocarcinoma", "OncoTree", "Condition", None],
            [45768916, "254626006", "Adenocarcinoma of lung", "SNOMED", "Condition", "S"],
        ],
    )
    con.execute("CREATE SCHEMA study_a")
    con.execute(
        "CREATE TABLE study_a.sample "
        "(ONCOTREE_CODE VARCHAR, CANCER_TYPE VARCHAR, CANCER_TYPE_DETAILED VARCHAR)"
    )
    con.executemany(
        "INSERT INTO study_a.sample VALUES (?, ?, ?)",
        [
            ["LUAD", "Non-Small Cell Lung Cancer", "Lung Adenocarcinoma"],
            ["COAD", "Colorectal Cancer", "Colon Adenocarcinoma"],
            ["ODD", None, None],
        ],
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


def test_the_worksheet_carries_the_main_type_a_reviewer_needs(
    gold_root: Path, db: str, tmp_path: Path
) -> None:
    """D4's anti-anchoring rule requires source-side context, since candidate
    targets are the resolver's own answers. A code and a name is not enough."""
    output_path = tmp_path / "ws.csv"

    _run("goldset", "worksheet", "--db", db, "--head-size", "2", "--output", str(output_path))
    by_code = {
        r["oncotree_code"]: r for r in csv.DictReader(output_path.open(encoding="utf-8"))
    }

    assert by_code["LUAD"]["main_type"] == "Non-Small Cell Lung Cancer"
    assert by_code["COAD"]["main_type"] == "Colorectal Cancer"


def test_the_worksheet_takes_tissue_from_the_reference_csv(
    gold_root: Path, db: str, tmp_path: Path
) -> None:
    (gold_root / "oncotree_reference_test.csv").write_text(
        "oncotree_code,name,mainType,tissue\nLUAD,Lung Adenocarcinoma,x,Lung\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "ws.csv"

    _run("goldset", "worksheet", "--db", db, "--head-size", "2", "--output", str(output_path))
    by_code = {
        r["oncotree_code"]: r for r in csv.DictReader(output_path.open(encoding="utf-8"))
    }

    assert by_code["LUAD"]["tissue"] == "Lung"


def test_the_worksheet_names_the_context_it_could_not_fill(
    gold_root: Path, db: str, tmp_path: Path
) -> None:
    """A silently blank column reads as 'no context exists', not 'not looked up'."""
    output = _run(
        "goldset", "worksheet", "--db", db, "--head-size", "2",
        "--output", str(tmp_path / "ws.csv"),
    )

    assert "tissue" in output
    assert "3" in output, "all three codes lack a tissue with no reference CSV present"


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
                "snapshot_version": "v1",
                "gold_label": "NO_MAP",
                "curator": "dean",
                "review_date": "2026-09-01",
                "evidence": "no acceptable Condition target in SNOMED@omop-vocab-2024",
                "notes": "",
            }
        )

    output = _run(
        "goldset", "apply-labels", "--worksheet", str(worksheet), "--db", db,
        "--version", "v2", "--date", "2026-09-01",
    )
    snapshot = load_snapshot(snapshot_dir("v2", gold_root))

    assert snapshot.by_code()["ODD"].gold_label is GoldLabel.NO_MAP
    assert "second reviewer" in output, "the unadjudicated gap is surfaced"
    assert "1 of 1 declared challenge codes" in output


# --- the target-side pin is enforced, not merely recorded --------------------


@pytest.fixture()
def store(tmp_path: Path) -> str:
    from sema.resolve.value_mapping_store import ValueMappingStore
    from sema.resolve.value_mapping_store_utils import ResolutionStatus, ValueMapping

    path = tmp_path / "store.duckdb"
    con = duckdb.connect(str(path))
    mapping_store = ValueMappingStore(con)
    mapping_store.upsert(
        [
            ValueMapping(
                source_vocabulary="OncoTree",
                normalized_source_value=code,
                target_property_ref="target.stage.condition_concept_id",
                target_field="condition_concept_id",
                vocab_binding="binding.condition",
                concept_id=concept,
                vocab_release="omop-vocab-2024",
                valid_start=None,
                valid_end=None,
                resolution_status=ResolutionStatus.RESOLVED,
                no_map_reason=None,
                confidence=1.0,
                status=Status.auto_accepted,
                resolver_policy_ref="omop.oncotree_condition",
                run_id="resolver-run-7",
            )
            for code, concept in (("LUAD", 45768916), ("COAD", 4180790))
        ]
    )
    mapping_store.close()
    return str(path)


def _report_args(store_path: str, out: Path, release: str) -> list[str]:
    return [
        "mapping-report", "--store", store_path,
        "--source-vocabulary", "OncoTree",
        "--target-property-ref", "target.stage.condition_concept_id",
        "--resolver-policy-ref", "omop.oncotree_condition",
        "--vocab-release", release,
        "--run-id", "run-1", "--output-dir", str(out),
    ]


def test_mapping_report_refuses_a_release_the_snapshot_does_not_pin(
    gold_root: Path, store: str, tmp_path: Path
) -> None:
    """A gold_concept_id is meaningless without the release that minted it, so
    grading a 2025 decision set on 2024 keys reads vocabulary churn as error."""
    result = CliRunner().invoke(
        eval_group, _report_args(store, tmp_path / "runs", "omop-vocab-2025")
    )

    assert result.exit_code != 0
    assert "omop-vocab-2025" in result.output
    assert "omop-vocab-2024" in result.output


def test_mapping_report_records_both_target_pins(
    gold_root: Path, store: str, tmp_path: Path
) -> None:
    out = tmp_path / "runs"
    _run(*_report_args(store, out, "omop-vocab-2024"))

    manifest = json.loads((out / "run-1" / "run.json").read_text(encoding="utf-8"))

    assert manifest["gold_set"]["vocab_release"] == "omop-vocab-2024"
    assert manifest["gold_set"]["target_vocabulary"] == "SNOMED"
    assert manifest["gold_set"]["target_domain"] == "Condition"
    assert manifest["resolver_run_ids"] == ["resolver-run-7"]


def _worksheet_file(tmp_path: Path, **row: str) -> Path:
    worksheet = tmp_path / "bad.csv"
    with worksheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(WORKSHEET_COLUMNS))
        writer.writeheader()
        writer.writerow(row)
    return worksheet


def _apply(worksheet: Path, db: str) -> object:
    return CliRunner().invoke(
        eval_group,
        ["goldset", "apply-labels", "--worksheet", str(worksheet), "--db", db,
         "--version", "v2", "--date", "2026-09-01"],
    )


def test_apply_labels_refuses_a_label_without_provenance(
    gold_root: Path, db: str, tmp_path: Path
) -> None:
    worksheet = _worksheet_file(
        tmp_path, oncotree_code="LUAD", snapshot_version="v1", gold_label="NO_MAP"
    )

    result = _apply(worksheet, db)

    assert result.exit_code != 0
    assert "annotation floor" in str(result.exception)


def test_apply_labels_checks_the_concept_the_label_names(
    gold_root: Path, db: str, tmp_path: Path
) -> None:
    """A transposed digit must be an error, not an oracle."""
    worksheet = _worksheet_file(
        tmp_path,
        oncotree_code="LUAD",
        snapshot_version="v1",
        gold_label="RESOLVED",
        gold_concept_id="45768916",
        target_concept_code="254626000",
        curator="dean",
        review_date="2026-09-01",
        evidence="SNOMED browser",
    )

    result = _apply(worksheet, db)

    assert result.exit_code != 0
    assert "254626006" in str(result.exception)


def test_apply_labels_accepts_a_label_that_matches_the_vocabulary(
    gold_root: Path, db: str, tmp_path: Path
) -> None:
    worksheet = _worksheet_file(
        tmp_path,
        oncotree_code="LUAD",
        snapshot_version="v1",
        gold_label="RESOLVED",
        gold_concept_id="45768916",
        target_concept_code="254626006",
        curator="dean",
        review_date="2026-09-01",
        evidence="SNOMED browser",
    )

    result = _apply(worksheet, db)

    assert result.exit_code == 0, result.output + str(result.exception)
    assert load_snapshot(snapshot_dir("v2", gold_root)).by_code()["LUAD"].gold_concept_id == 45768916
