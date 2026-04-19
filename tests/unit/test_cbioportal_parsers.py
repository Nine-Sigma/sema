from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sema.ingest.cbioportal import (
    fetch_study_files,
    iter_timeline_files,
    parse_clinical_file,
    parse_clinical_header,
    parse_maf,
    parse_timeline_file,
)


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.unit
class TestClinicalHeaderParser:
    def test_parses_five_line_metadata_block(self) -> None:
        lines = [
            "#Patient Identifier\tAge at Diagnosis",
            "#Identifier\tAge",
            "#STRING\tNUMBER",
            "#1\t2",
            "#PATIENT_ID\tAGE",
            "PATIENT_ID\tAGE",
        ]
        header = parse_clinical_header(lines)
        assert header.column_names == ["PATIENT_ID", "AGE"]
        assert header.display_names == ["Patient Identifier", "Age at Diagnosis"]
        assert header.descriptions == ["Identifier", "Age"]
        assert header.types == ["STRING", "NUMBER"]

    def test_parses_four_line_metadata_block(self) -> None:
        lines = [
            "#Patient Identifier\tAge",
            "#Patient ID\tAge in years",
            "#STRING\tNUMBER",
            "#PATIENT_ID\tAGE",
            "PATIENT_ID\tAGE",
        ]
        header = parse_clinical_header(lines)
        assert header.column_names == ["PATIENT_ID", "AGE"]
        assert header.types == ["STRING", "NUMBER"]

    def test_parses_three_line_metadata_block(self) -> None:
        lines = [
            "#Patient Identifier\tAge",
            "#STRING\tNUMBER",
            "#PATIENT_ID\tAGE",
            "PATIENT_ID\tAGE",
        ]
        header = parse_clinical_header(lines)
        assert header.column_names == ["PATIENT_ID", "AGE"]

    def test_returns_header_with_empty_metadata_when_no_prefix_lines(self) -> None:
        lines = ["PATIENT_ID\tAGE"]
        header = parse_clinical_header(lines)
        assert header.column_names == ["PATIENT_ID", "AGE"]
        assert header.types == []


@pytest.mark.unit
class TestParseClinicalFile:
    def test_parses_file_with_rows_and_column_comments(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_clinical_patient.txt",
            "#Patient Identifier\tAge\n"
            "#Patient ID\tAge at diagnosis\n"
            "#STRING\tNUMBER\n"
            "#1\t1\n"
            "PATIENT_ID\tAGE\n"
            "P-001\t42\n"
            "P-002\t55\n",
        )
        rows, column_types, column_comments = parse_clinical_file(path)
        assert rows.num_rows == 2
        assert rows.column_names == ["PATIENT_ID", "AGE"]
        assert column_comments["AGE"] == "Age at diagnosis"
        assert column_types["AGE"] in {"DOUBLE", "BIGINT", "INTEGER"}

    def test_skips_malformed_row_and_warns(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_clinical_sample.txt",
            "#Sample ID\tStudy\n"
            "#Sample id\tStudy name\n"
            "#STRING\tSTRING\n"
            "#1\t1\n"
            "SAMPLE_ID\tSTUDY\n"
            "S-1\tbrca\n"
            "S-2\n"
            "S-3\tbrca\n",
        )
        rows, _types, _comments = parse_clinical_file(path)
        assert rows.num_rows == 2

    def test_decodes_invalid_utf8_with_replacement(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.txt"
        path.write_bytes(
            b"#ID\tNote\n"
            b"#Sample ID\tNote\n"
            b"#STRING\tSTRING\n"
            b"#1\t1\n"
            b"ID\tNOTE\n"
            b"S-1\tcaf\xe9\n"
        )
        rows, _, _ = parse_clinical_file(path)
        assert rows.num_rows == 1


@pytest.mark.unit
class TestParseMAF:
    def test_parses_maf_preserving_all_columns(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_mutations_extended.txt",
            "Hugo_Symbol\tChromosome\tStart_Position\tEnd_Position\tCustom_Field\n"
            "TP53\t17\t7571720\t7590868\tnote\n"
            "BRCA1\t17\t41196311\t41277500\textra\n",
        )
        rows, column_types, _ = parse_maf(path)
        assert rows.num_rows == 2
        assert "Custom_Field" in rows.column_names
        assert column_types["Start_Position"] in {"BIGINT", "INTEGER"}
        assert column_types["Custom_Field"] == "VARCHAR"
        assert column_types["Hugo_Symbol"] == "VARCHAR"

    def test_skips_comment_prefixed_lines(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "maf_with_comment.txt",
            "#version 2.4\n"
            "Hugo_Symbol\tStart_Position\n"
            "TP53\t7571720\n",
        )
        rows, _, _ = parse_maf(path)
        assert rows.num_rows == 1


@pytest.mark.unit
class TestTimelineParsing:
    def test_parse_timeline_file_produces_table(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "data_timeline_treatment.txt",
            "PATIENT_ID\tSTART_DATE\tSTOP_DATE\tEVENT_TYPE\n"
            "P-001\t0\t30\tTREATMENT\n",
        )
        rows, column_types, _ = parse_timeline_file(path)
        assert rows.num_rows == 1
        assert "EVENT_TYPE" in rows.column_names

    def test_iter_timeline_files_emits_one_kind_per_file(self, tmp_path: Path) -> None:
        _write(tmp_path / "data_timeline_treatment.txt", "PATIENT_ID\n")
        _write(tmp_path / "data_timeline_status.txt", "PATIENT_ID\n")
        _write(tmp_path / "data_clinical_patient.txt", "PATIENT_ID\n")

        kinds = {kind: path for kind, path in iter_timeline_files(tmp_path)}
        assert set(kinds.keys()) == {"treatment", "status"}
        assert "timeline_treatment" not in kinds

    def test_iter_timeline_files_returns_empty_when_no_files(self, tmp_path: Path) -> None:
        assert list(iter_timeline_files(tmp_path)) == []


@pytest.mark.unit
class TestIngestStudySkipsMatrixFiles:
    def test_skips_expression_methylation_and_case_lists(
        self, tmp_path: Path,
    ) -> None:
        from sema.ingest.cbioportal import _list_skipped_files

        (tmp_path / "data_expression_median.txt").write_text("")
        (tmp_path / "data_methylation_hm27.txt").write_text("")
        (tmp_path / "data_log2_cna.txt").write_text("")
        (tmp_path / "case_lists").mkdir()
        (tmp_path / "data_clinical_patient.txt").write_text("")
        (tmp_path / "data_cna.txt").write_text("")

        skipped = _list_skipped_files(tmp_path)
        names = {p.name for p in skipped}
        assert "data_expression_median.txt" in names
        assert "data_methylation_hm27.txt" in names
        assert "data_log2_cna.txt" in names
        assert "data_clinical_patient.txt" not in names
        assert "data_cna.txt" not in names


@pytest.mark.unit
class TestFetchStudyFiles:
    def test_reuses_cache_when_done_marker_present(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        study_dir = cache / "brca_tcga"
        study_dir.mkdir(parents=True)
        (study_dir / ".done").touch()

        with patch("sema.ingest.cbioportal.urlopen") as mock_urlopen:
            result = fetch_study_files("brca_tcga", cache_dir=cache)
            mock_urlopen.assert_not_called()
        assert result == study_dir

    def test_lists_and_downloads_expected_files(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        api_entries = [
            {"name": "data_clinical_patient.txt", "type": "file", "download_url": "https://example/dl/data_clinical_patient.txt"},
            {"name": "data_clinical_sample.txt", "type": "file", "download_url": "https://example/dl/data_clinical_sample.txt"},
            {"name": "data_mutations.txt", "type": "file", "download_url": "https://example/dl/data_mutations.txt"},
            {"name": "data_timeline_treatment.txt", "type": "file", "download_url": "https://example/dl/data_timeline_treatment.txt"},
            {"name": "data_CNA.txt", "type": "file", "download_url": "https://example/dl/data_CNA.txt"},
            {"name": "data_expression_median.txt", "type": "file", "download_url": "https://example/dl/data_expression_median.txt"},
            {"name": "case_lists", "type": "dir"},
            {"name": "README.md", "type": "file", "download_url": "https://example/dl/README.md"},
            {"name": "meta_study.txt", "type": "file", "download_url": "https://example/dl/meta_study.txt"},
        ]
        api_resp = MagicMock()
        api_resp.read.return_value = json.dumps(api_entries).encode("utf-8")
        api_resp.__enter__ = lambda self: self
        api_resp.__exit__ = lambda self, *a: None

        download_responses: list[MagicMock] = []
        for _ in api_entries:
            dl = MagicMock()
            dl.read.side_effect = [b"payload", b""]
            dl.__enter__ = lambda self: self
            dl.__exit__ = lambda self, *a: None
            download_responses.append(dl)

        urlopen_mock = MagicMock(side_effect=[api_resp, *download_responses])
        with patch("sema.ingest.cbioportal.urlopen", urlopen_mock):
            result = fetch_study_files("brca_tcga", cache_dir=cache)

        downloaded = {p.name for p in result.iterdir() if p.is_file() and p.name != ".done"}
        assert "data_clinical_patient.txt" in downloaded
        assert "data_clinical_sample.txt" in downloaded
        assert "data_mutations.txt" in downloaded
        assert "data_timeline_treatment.txt" in downloaded
        assert "meta_study.txt" in downloaded
        assert "data_CNA.txt" not in downloaded
        assert "data_expression_median.txt" not in downloaded
        assert "README.md" not in downloaded
        assert (result / ".done").exists()


