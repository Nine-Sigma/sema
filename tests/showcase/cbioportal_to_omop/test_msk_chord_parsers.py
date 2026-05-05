"""Tests for MSK CHORD-specific parsers: segmented CNA, gene panel matrix (long), lab timelines."""
from __future__ import annotations

from pathlib import Path

import pytest

from showcase.cbioportal_to_omop.parsers import (
    parse_gene_panel_matrix,
    parse_lab_timeline,
    parse_segmented_cna,
    parse_timeline_file,
)

pytestmark = pytest.mark.unit


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestParseSegmentedCna:
    def test_parses_six_column_seg_file(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_cna_hg19.seg",
            "ID\tchrom\tloc.start\tloc.end\tnum.mark\tseg.mean\n"
            "P-001\t1\t10000\t250000\t150\t0.4\n"
            "P-001\t2\t300000\t500000\t80\t-0.2\n"
            "P-002\t1\t10000\t250000\t150\t0.0\n",
        )
        rows, types, _ = parse_segmented_cna(path)
        assert rows.num_rows == 3
        col_set = set(rows.column_names)
        assert col_set == {"sample_id", "chrom", "loc_start", "loc_end", "num_mark", "seg_mean"}

    def test_numeric_columns_typed_correctly(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "x.seg",
            "ID\tchrom\tloc.start\tloc.end\tnum.mark\tseg.mean\n"
            "S-1\tX\t1\t1000\t10\t0.3\n",
        )
        _, types, _ = parse_segmented_cna(path)
        assert types["loc_start"] == "BIGINT"
        assert types["loc_end"] == "BIGINT"
        assert types["num_mark"] == "BIGINT"
        assert types["seg_mean"] == "DOUBLE"
        assert types["sample_id"] == "VARCHAR"
        assert types["chrom"] == "VARCHAR"

    def test_handles_blank_seg_mean_as_null(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "x.seg",
            "ID\tchrom\tloc.start\tloc.end\tnum.mark\tseg.mean\n"
            "S-1\t1\t1\t1000\t10\t\n",
        )
        rows, _, _ = parse_segmented_cna(path)
        assert rows.column("seg_mean").to_pylist() == [None]


class TestParseGenePanelMatrixLong:
    def test_pivots_wide_to_long(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_gene_panel_matrix.txt",
            "SAMPLE_ID\tmutations\tcna\tstructural_variants\n"
            "S-1\tIMPACT341\tIMPACT341\tIMPACT341\n"
            "S-2\tIMPACT410\tIMPACT410\tIMPACT410\n",
        )
        rows, types, _ = parse_gene_panel_matrix(path)
        col_set = set(rows.column_names)
        assert col_set == {"sample_id", "panel_id", "assay"}
        # 2 samples x 3 assays = 6 rows
        assert rows.num_rows == 6
        assert types["sample_id"] == "VARCHAR"
        assert types["panel_id"] == "VARCHAR"
        assert types["assay"] == "VARCHAR"

    def test_skips_blank_panel_assignments(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_gene_panel_matrix.txt",
            "SAMPLE_ID\tmutations\tcna\n"
            "S-1\tIMPACT341\t\n"
            "S-2\t\tIMPACT410\n",
        )
        rows, _, _ = parse_gene_panel_matrix(path)
        # Blank cells skipped: 2 non-blank assignments only
        assert rows.num_rows == 2
        pairs = list(zip(
            rows.column("sample_id").to_pylist(),
            rows.column("assay").to_pylist(),
        ))
        assert ("S-1", "mutations") in pairs
        assert ("S-2", "cna") in pairs

    def test_emits_one_row_per_sample_assay_pair(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_gene_panel_matrix.txt",
            "SAMPLE_ID\tmutations\n"
            "S-1\tIMPACT341\n",
        )
        rows, _, _ = parse_gene_panel_matrix(path)
        assert rows.num_rows == 1
        assert rows.column("panel_id").to_pylist() == ["IMPACT341"]
        assert rows.column("assay").to_pylist() == ["mutations"]


class TestParseLabTimeline:
    def test_types_value_as_double_when_test_value_units_present(
        self, tmp_path: Path
    ) -> None:
        path = _write(
            tmp_path / "data_timeline_labtest.txt",
            "PATIENT_ID\tSTART_DATE\tSTOP_DATE\tEVENT_TYPE\tTEST\tVALUE\tUNITS\n"
            "P-1\t10\t10\tLAB_TEST\tHemoglobin\t13.5\tg/dL\n"
            "P-2\t20\t20\tLAB_TEST\tCreatinine\t1.1\tmg/dL\n",
        )
        rows, types, _ = parse_lab_timeline(path)
        assert types["VALUE"] == "DOUBLE"
        assert types["TEST"] == "VARCHAR"
        assert types["UNITS"] == "VARCHAR"
        assert rows.column("VALUE").to_pylist() == [13.5, 1.1]

    def test_falls_back_to_varchar_when_value_unparseable(
        self, tmp_path: Path
    ) -> None:
        path = _write(
            tmp_path / "data_timeline_labtest.txt",
            "PATIENT_ID\tSTART_DATE\tEVENT_TYPE\tTEST\tVALUE\tUNITS\n"
            "P-1\t10\tLAB_TEST\tA1c\t<6.0\t%\n",
        )
        rows, types, _ = parse_lab_timeline(path)
        assert types["VALUE"] == "DOUBLE"
        assert rows.column("VALUE").to_pylist() == [None]


class TestParseTimelineFileGeneric:
    def test_non_lab_timeline_keeps_varchar(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_timeline_treatment.txt",
            "PATIENT_ID\tSTART_DATE\tSTOP_DATE\tEVENT_TYPE\tTREATMENT_TYPE\tAGENT\n"
            "P-1\t0\t30\tTREATMENT\tChemotherapy\tCisplatin\n",
        )
        _, types, _ = parse_timeline_file(path)
        assert types["AGENT"] == "VARCHAR"


class TestIngestStudyDirDispatchesSeg:
    def test_seg_file_creates_cna_segmented_table(self, tmp_path: Path) -> None:
        from sema.ingest.duckdb_staging import Staging
        from showcase.cbioportal_to_omop.parsers import _ingest_study_dir

        study_dir = tmp_path / "msk_chord"
        study_dir.mkdir()
        _write(
            study_dir / "msk_chord_2024_data_cna_hg19.seg",
            "ID\tchrom\tloc.start\tloc.end\tnum.mark\tseg.mean\n"
            "S-1\t1\t1\t1000\t10\t0.3\n",
        )

        staging = Staging(
            db_path=str(tmp_path / "db.duckdb"),
            schemas=("cbioportal",),
        )
        _ingest_study_dir("msk_chord", study_dir, staging, schema_name="cbioportal")
        info = staging.describe("cbioportal", "cna_segmented")
        assert "sample_id" in info.columns
        assert "seg_mean" in info.columns
        staging.close()

    def test_lab_timeline_picked_up_by_dispatch(self, tmp_path: Path) -> None:
        from sema.ingest.duckdb_staging import Staging
        from showcase.cbioportal_to_omop.parsers import _ingest_study_dir

        study_dir = tmp_path / "msk_chord"
        study_dir.mkdir()
        _write(
            study_dir / "data_timeline_labtest.txt",
            "PATIENT_ID\tSTART_DATE\tEVENT_TYPE\tTEST\tVALUE\tUNITS\n"
            "P-1\t10\tLAB_TEST\tHb\t13.5\tg/dL\n",
        )

        staging = Staging(
            db_path=str(tmp_path / "db.duckdb"),
            schemas=("cbioportal",),
        )
        _ingest_study_dir("msk_chord", study_dir, staging, schema_name="cbioportal")
        info = staging.describe("cbioportal", "timeline_labtest")
        assert info.columns["VALUE"].type.upper().startswith("DOUB")
        staging.close()


class TestRoundTripMixedFiles:
    def test_ingest_mixed_seg_and_panel_matrix_and_lab(
        self, tmp_path: Path
    ) -> None:
        from sema.ingest.duckdb_staging import Staging
        from showcase.cbioportal_to_omop.parsers import _ingest_study_dir

        study_dir = tmp_path / "study"
        study_dir.mkdir()
        _write(
            study_dir / "msk_data_cna_hg19.seg",
            "ID\tchrom\tloc.start\tloc.end\tnum.mark\tseg.mean\nS-1\t1\t1\t100\t5\t0.1\n",
        )
        _write(
            study_dir / "data_gene_panel_matrix.txt",
            "SAMPLE_ID\tmutations\nS-1\tIMPACT341\n",
        )
        _write(
            study_dir / "data_timeline_labtest.txt",
            "PATIENT_ID\tSTART_DATE\tEVENT_TYPE\tTEST\tVALUE\tUNITS\n"
            "P-1\t10\tLAB_TEST\tHb\t13.5\tg/dL\n",
        )

        staging = Staging(
            db_path=str(tmp_path / "rt.duckdb"),
            schemas=("cbioportal",),
        )
        _ingest_study_dir("msk", study_dir, staging, schema_name="cbioportal")

        for tbl in ("cna_segmented", "gene_panel_matrix", "timeline_labtest"):
            info = staging.describe("cbioportal", tbl)
            assert info.columns, f"{tbl} should be ingested"
        # gene_panel_matrix should be long format
        gpm = staging.describe("cbioportal", "gene_panel_matrix")
        assert "panel_id" in gpm.columns
        assert "assay" in gpm.columns
        staging.close()
