"""Tests for the eval slice runner."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sema.eval.runner import (
    build_diff_report,
    build_run_report,
    load_slice,
    pair_dumps_by_table,
    write_table_dump,
    write_telemetry_dump,
)
from sema.eval.runner_utils import SliceDefinition

pytestmark = pytest.mark.unit


@pytest.fixture
def slice_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "slice.yaml"
    path.write_text(yaml.safe_dump({
        "version": 1,
        "catalog": "unity",
        "schema": "cbioportal",
        "tables": [
            {"table_name": "patient", "tier": "sanity"},
            {"table_name": "sample", "tier": "sanity"},
            {"table_name": "mutation", "tier": "stress"},
        ],
    }))
    return path


class TestLoadSlice:
    def test_parses_catalog_schema_and_tables(self, slice_yaml: Path) -> None:
        sdef = load_slice(slice_yaml)
        assert sdef.catalog == "unity"
        assert sdef.schema == "cbioportal"
        assert sdef.tables == ["patient", "sample", "mutation"]

    def test_preserves_table_metadata(self, slice_yaml: Path) -> None:
        sdef = load_slice(slice_yaml)
        assert sdef.table_meta["patient"]["tier"] == "sanity"
        assert sdef.table_meta["mutation"]["tier"] == "stress"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_slice(tmp_path / "missing.yaml")

    def test_raises_on_empty_tables(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text(yaml.safe_dump({
            "catalog": "c", "schema": "s", "tables": [],
        }))
        with pytest.raises(ValueError, match="no tables"):
            load_slice(path)


class TestWriteTableDump:
    def test_writes_assertion_dump_json(self, tmp_path: Path) -> None:
        from sema.models.assertions import (
            Assertion,
            AssertionPredicate,
        )
        from datetime import datetime, timezone

        a = Assertion(
            id="x", subject_ref="unity://c/s/t",
            predicate=AssertionPredicate.HAS_ENTITY_NAME,
            payload={"value": "Patient"},
            source="llm_interpretation",
            confidence=0.9,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
        )
        out = write_table_dump(
            [a], table_ref="unity://c/s/t",
            label="staged", output_dir=tmp_path,
        )
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["config_label"] == "staged"
        assert len(loaded["assertions"]) == 1
        assert loaded["assertions"][0]["payload"]["value"] == "Patient"

    def test_filename_is_deterministic_per_table_and_label(
        self, tmp_path: Path,
    ) -> None:
        out = write_table_dump(
            [], table_ref="unity://c/s/patient",
            label="baseline", output_dir=tmp_path,
        )
        assert out.name == "patient__baseline.json"


class TestWriteTelemetryDump:
    def test_writes_telemetry_json(self, tmp_path: Path) -> None:
        telemetry = {
            "table_ref": "unity://c/s/patient",
            "b_outcome": "B_SUCCESS",
            "raw_coverage_pct": 1.0,
        }
        out = write_telemetry_dump(
            telemetry, table_ref="unity://c/s/patient",
            label="staged", output_dir=tmp_path,
        )
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["b_outcome"] == "B_SUCCESS"

    def test_filename_includes_telemetry_suffix(
        self, tmp_path: Path,
    ) -> None:
        out = write_telemetry_dump(
            {}, table_ref="unity://c/s/patient",
            label="staged", output_dir=tmp_path,
        )
        assert out.name == "patient__staged__telemetry.json"


class TestPairDumpsByTable:
    def test_pairs_matching_tables(self, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline"
        current = tmp_path / "current"
        baseline.mkdir()
        current.mkdir()
        (baseline / "patient__baseline.json").write_text("{}")
        (baseline / "sample__baseline.json").write_text("{}")
        (current / "patient__staged.json").write_text("{}")
        (current / "mutation__staged.json").write_text("{}")

        pairs, unmatched = pair_dumps_by_table(baseline, current)
        paired_tables = {p[0] for p in pairs}
        assert paired_tables == {"patient"}
        assert "sample" in unmatched["only_in_baseline"]
        assert "mutation" in unmatched["only_in_current"]

    def test_ignores_telemetry_files_when_pairing(
        self, tmp_path: Path,
    ) -> None:
        baseline = tmp_path / "baseline"
        current = tmp_path / "current"
        baseline.mkdir()
        current.mkdir()
        (baseline / "patient__x.json").write_text("{}")
        (baseline / "patient__x__telemetry.json").write_text("{}")
        (current / "patient__y.json").write_text("{}")

        pairs, _ = pair_dumps_by_table(baseline, current)
        assert len(pairs) == 1


class TestBuildDiffReport:
    def test_aggregates_diffs_across_tables(
        self, tmp_path: Path,
    ) -> None:
        baseline = tmp_path / "b"
        current = tmp_path / "c"
        baseline.mkdir()
        current.mkdir()
        _write_dump(
            baseline / "patient__b.json",
            assertions=[{
                "subject_ref": "unity://c/s/patient",
                "predicate": "has_entity_name",
                "payload": {"value": "Old"},
                "confidence": 0.8, "source": "llm",
            }],
        )
        _write_dump(
            current / "patient__c.json",
            assertions=[{
                "subject_ref": "unity://c/s/patient",
                "predicate": "has_entity_name",
                "payload": {"value": "New"},
                "confidence": 0.9, "source": "llm",
            }],
        )

        report = build_diff_report(baseline, current)
        assert report["summary"]["tables_compared"] == 1
        assert report["summary"]["total_changed"] == 1


class TestBuildRunReport:
    def test_aggregates_telemetry_from_dumps(
        self, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_dump(
            run_dir / "patient__x__telemetry.json",
            payload={
                "table_ref": "unity://c/s/patient",
                "b_outcome": "B_SUCCESS",
                "raw_coverage_pct": 1.0,
                "critical_coverage_pct": 1.0,
                "c_trigger_rate": 0.2, "total_latency_ms": 4000,
                "retries_used": 0, "splits_used": 0,
                "rescues_used": 0, "tokens_input": 100,
                "tokens_output": 50, "stage_c_calls": 2,
            },
        )

        report = build_run_report(run_dir, label="staged")
        assert report["label"] == "staged"
        assert report["telemetry"]["table_count"] == 1


def _write_dump(path: Path, assertions: list | None = None,
                payload: dict | None = None) -> None:
    if payload is None:
        payload = {
            "table_ref": "unity://c/s/patient",
            "config_label": "x",
            "timestamp": "2026-04-19T00:00:00+00:00",
            "run_id": None,
            "assertions": assertions or [],
        }
    path.write_text(json.dumps(payload))
