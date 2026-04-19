"""Tests for extended cBioPortal parsers: SV, CNA, gene panel matrix, resources."""
from __future__ import annotations

from pathlib import Path

import pytest

from sema.ingest.cbioportal import (
    parse_cna_file,
    parse_gene_panel_matrix,
    parse_resource_file,
    parse_sv_file,
)

pytestmark = pytest.mark.unit


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestParseSVFile:
    def test_parses_sv_tsv_preserving_columns(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_sv.txt",
            "Sample_Id\tSite1_Hugo_Symbol\tSite1_Entrez_Gene_Id\t"
            "Site1_Position\tSite2_Hugo_Symbol\tSite2_Position\t"
            "SV_Status\tClass\n"
            "SAMPLE-1\tEML4\t27436\t42491877\tALK\t29455586\tSOMATIC\tFUSION\n"
            "SAMPLE-2\tBCR\t613\t23632600\tABL1\t133738363\tSOMATIC\tFUSION\n",
        )
        rows, types, _ = parse_sv_file(path)
        assert rows.num_rows == 2
        assert "Site1_Hugo_Symbol" in rows.column_names
        assert "SV_Status" in rows.column_names

    def test_numeric_position_columns_typed_as_bigint(
        self, tmp_path: Path,
    ) -> None:
        path = _write(
            tmp_path / "data_sv.txt",
            "Sample_Id\tSite1_Position\tSite2_Position\tClass\n"
            "S-1\t42491877\t29455586\tFUSION\n",
        )
        _, types, _ = parse_sv_file(path)
        assert types["Site1_Position"] == "BIGINT"
        assert types["Site2_Position"] == "BIGINT"
        assert types["Class"] == "VARCHAR"

    def test_handles_comment_prefixed_lines(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_sv.txt",
            "#version 1.0\n"
            "Sample_Id\tClass\n"
            "S-1\tFUSION\n",
        )
        rows, _, _ = parse_sv_file(path)
        assert rows.num_rows == 1


class TestParseCNAFile:
    def test_pivots_wide_matrix_to_long_format(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_cna.txt",
            "Hugo_Symbol\tEntrez_Gene_Id\tTCGA-02-0001\tTCGA-02-0003\n"
            "EGFR\t1956\t2\t-1\n"
            "TP53\t7157\t0\t-2\n",
        )
        rows, types, _ = parse_cna_file(path)
        assert rows.num_rows == 4
        col_names = set(rows.column_names)
        assert col_names == {
            "sample_id", "hugo_symbol", "entrez_gene_id", "cna_value",
        }
        assert types["cna_value"] == "INTEGER"
        assert types["entrez_gene_id"] == "BIGINT"
        assert types["sample_id"] == "VARCHAR"

    def test_skips_blank_values_in_matrix(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_cna.txt",
            "Hugo_Symbol\tEntrez_Gene_Id\tSAMPLE-1\tSAMPLE-2\n"
            "EGFR\t1956\t\t-1\n",
        )
        rows, _, _ = parse_cna_file(path)
        values = rows.column("cna_value").to_pylist()
        samples = rows.column("sample_id").to_pylist()
        assert rows.num_rows == 2
        assert ("SAMPLE-1", None) in list(zip(samples, values))
        assert ("SAMPLE-2", -1) in list(zip(samples, values))

    def test_handles_file_without_entrez_column(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_cna.txt",
            "Hugo_Symbol\tSAMPLE-1\n"
            "EGFR\t2\n",
        )
        rows, _, _ = parse_cna_file(path)
        assert rows.num_rows == 1
        assert "entrez_gene_id" in rows.column_names


class TestParseGenePanelMatrix:
    def test_parses_sample_to_panel_assignments(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_gene_panel_matrix.txt",
            "SAMPLE_ID\tmutations\tcna\tstructural_variants\n"
            "SAMPLE-1\tIMPACT341\tIMPACT341\tIMPACT341\n"
            "SAMPLE-2\tIMPACT410\tIMPACT410\tIMPACT410\n",
        )
        rows, types, _ = parse_gene_panel_matrix(path)
        assert rows.num_rows == 2
        assert "SAMPLE_ID" in rows.column_names
        assert types["mutations"] == "VARCHAR"


class TestParseResourceFile:
    def test_parses_resource_definition(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_resource_definition.txt",
            "RESOURCE_ID\tDISPLAY_NAME\tDESCRIPTION\tRESOURCE_TYPE\tOPEN_BY_DEFAULT\tPRIORITY\n"
            "imaging\tMRI Scan\tT1-weighted MRI\tPATIENT\tfalse\t1\n"
            "pathology\tPath Report\tHistology\tPATIENT\tfalse\t2\n",
        )
        rows, types, _ = parse_resource_file(path)
        assert rows.num_rows == 2
        assert "RESOURCE_ID" in rows.column_names
        assert types["PRIORITY"] == "VARCHAR"

    def test_parses_resource_patient(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_resource_patient.txt",
            "PATIENT_ID\tRESOURCE_ID\tURL\n"
            "P-1\timaging\thttp://example.com/img1\n"
            "P-2\tpathology\thttp://example.com/path1\n",
        )
        rows, _, _ = parse_resource_file(path)
        assert rows.num_rows == 2
        assert "URL" in rows.column_names


class TestDownloadFilterIncludesNewFileTypes:
    def test_should_download_includes_new_types(self) -> None:
        from sema.ingest.cbioportal import _should_download

        for name in (
            "data_sv.txt",
            "data_cna.txt",
            "data_gene_panel_matrix.txt",
            "data_resource_definition.txt",
            "data_resource_patient.txt",
            "data_resource_sample.txt",
            "data_clinical_supp_hypoxia.txt",
        ):
            assert _should_download(name), f"should download {name}"

    def test_still_skips_non_ingested_types(self) -> None:
        from sema.ingest.cbioportal import _should_download

        for name in (
            "data_expression_median.txt",
            "data_methylation_hm27.txt",
            "data_log2_cna.txt",
            "data_mrna_seq_v2_rsem.txt",
            "README.md",
        ):
            assert not _should_download(name), f"should NOT download {name}"


class TestIngestStudyDirWiringNewTables:
    def test_ingests_sv_cna_panel_matrix_and_resources(
        self, tmp_path: Path,
    ) -> None:
        from sema.ingest.cbioportal import _ingest_study_dir
        from sema.ingest.duckdb_staging import Staging

        study_dir = tmp_path / "study"
        study_dir.mkdir()
        _write(study_dir / "data_sv.txt",
               "Sample_Id\tClass\nS-1\tFUSION\n")
        _write(study_dir / "data_cna.txt",
               "Hugo_Symbol\tEntrez_Gene_Id\tS-1\nEGFR\t1956\t2\n")
        _write(study_dir / "data_gene_panel_matrix.txt",
               "SAMPLE_ID\tmutations\nS-1\tIMPACT341\n")
        _write(study_dir / "data_resource_definition.txt",
               "RESOURCE_ID\tRESOURCE_TYPE\nimaging\tPATIENT\n")
        _write(study_dir / "data_resource_patient.txt",
               "PATIENT_ID\tRESOURCE_ID\tURL\nP-1\timaging\thttp://x\n")

        staging = Staging(
            db_path=str(tmp_path / "db.duckdb"),
            schemas=("cbioportal",),
        )
        _ingest_study_dir("test_study", study_dir, staging)
        for tbl in (
            "structural_variant", "cna", "gene_panel_matrix",
            "resource_definition", "resource_patient",
        ):
            info = staging.describe("cbioportal", tbl)
            assert info.columns, f"{tbl} should have columns"
        staging.close()
