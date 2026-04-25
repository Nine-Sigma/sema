"""Tests for slice contamination check: holdout must not overlap with
few-shot source tables."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.check_slice_contamination import (
    ContaminationError,
    check_contamination,
    load_contamination_map,
    load_slice_tables,
)

pytestmark = pytest.mark.unit


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class TestLoadContaminationMap:
    def test_loads_table_names(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path / "contamination_map.yaml",
            {"contaminated_tables": ["patient", "sample"]},
        )
        result = load_contamination_map(path)
        assert result == {"patient", "sample"}

    def test_returns_empty_set_when_field_missing(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path / "x.yaml", {"version": 1})
        assert load_contamination_map(path) == set()


class TestLoadSliceTables:
    def test_extracts_table_names(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path / "slice.yaml",
            {
                "schema": "cbioportal_msk_chord_2024",
                "tables": [
                    {"table_name": "timeline_x", "tier": "standard"},
                    {"table_name": "timeline_y", "tier": "edge"},
                ],
            },
        )
        result = load_slice_tables(path)
        assert result == {"timeline_x", "timeline_y"}

    def test_empty_tables_returns_empty_set(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path / "empty.yaml", {"tables": []})
        assert load_slice_tables(path) == set()


class TestCheckContamination:
    def test_no_overlap_passes(self, tmp_path: Path) -> None:
        contam = _write_yaml(
            tmp_path / "contam.yaml",
            {"contaminated_tables": ["patient", "sample"]},
        )
        holdout = _write_yaml(
            tmp_path / "holdout.yaml",
            {"tables": [{"table_name": "timeline_other"}]},
        )
        check_contamination(holdout, [contam])

    def test_overlap_raises(self, tmp_path: Path) -> None:
        contam = _write_yaml(
            tmp_path / "contam.yaml",
            {"contaminated_tables": ["patient"]},
        )
        holdout = _write_yaml(
            tmp_path / "holdout.yaml",
            {"tables": [{"table_name": "patient"}, {"table_name": "x"}]},
        )
        with pytest.raises(ContaminationError) as exc_info:
            check_contamination(holdout, [contam])
        assert "patient" in str(exc_info.value)

    def test_dev_slice_overlap_with_holdout_raises(self, tmp_path: Path) -> None:
        contam = _write_yaml(
            tmp_path / "contam.yaml",
            {"contaminated_tables": []},
        )
        dev = _write_yaml(
            tmp_path / "dev.yaml",
            {"tables": [{"table_name": "shared_table"}]},
        )
        holdout = _write_yaml(
            tmp_path / "holdout.yaml",
            {"tables": [{"table_name": "shared_table"}]},
        )
        with pytest.raises(ContaminationError) as exc_info:
            check_contamination(holdout, [contam, dev])
        assert "shared_table" in str(exc_info.value)


class TestProjectSliceFilesAreClean:
    def test_msk_chord_holdout_disjoint_from_few_shots_and_dev(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        slices = repo_root / "showcase" / "cbioportal_to_omop" / "slices"
        contam = slices / "contamination_map.yaml"
        dev = slices / "msk_chord_dev.yaml"
        holdout = slices / "msk_chord_holdout.yaml"
        check_contamination(holdout, [contam, dev])
