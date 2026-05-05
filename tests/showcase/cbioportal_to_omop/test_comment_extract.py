from __future__ import annotations

from pathlib import Path

import pytest

from showcase.cbioportal_to_omop.comment_extract import extract_study_comments

pytestmark = pytest.mark.unit


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_extract_clinical_patient_threads_column_and_table_comments(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "data_clinical_patient.txt",
        "#Patient Identifier\tAge at Diagnosis\n"
        "#Identifier to uniquely specify a patient.\tAge at first diagnosis.\n"
        "#STRING\tNUMBER\n"
        "#1\t1\n"
        "PATIENT_ID\tAGE\n"
        "P-001\t42\n",
    )
    out = extract_study_comments(tmp_path)
    assert "patient" in out
    patient = out["patient"]
    assert (
        patient.column_comments["PATIENT_ID"]
        == "Identifier to uniquely specify a patient."
    )
    assert patient.column_comments["AGE"] == "Age at first diagnosis."
    assert patient.table_comment == (
        "cBioPortal clinical patient from data_clinical_patient.txt"
    )


def test_extract_clinical_supp_files(tmp_path: Path) -> None:
    _write(
        tmp_path / "data_clinical_supp_hypoxia.txt",
        "#Hypoxia Score\n"
        "#Buffa hypoxia score.\n"
        "#NUMBER\n"
        "#1\n"
        "BUFFA_HYPOXIA_SCORE\n"
        "0.5\n",
    )
    out = extract_study_comments(tmp_path)
    assert "clinical_supp_hypoxia" in out
    assert (
        out["clinical_supp_hypoxia"].column_comments["BUFFA_HYPOXIA_SCORE"]
        == "Buffa hypoxia score."
    )


def test_extract_non_clinical_table_has_empty_column_comments(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "data_mutations.txt",
        "Hugo_Symbol\tStart_Position\n" "TP53\t7571720\n",
    )
    out = extract_study_comments(tmp_path)
    assert "mutation" in out
    assert out["mutation"].column_comments == {}
    assert out["mutation"].table_comment is not None
    assert "MAF" in out["mutation"].table_comment


def test_extract_returns_empty_dict_for_empty_dir(tmp_path: Path) -> None:
    assert extract_study_comments(tmp_path) == {}


def test_extract_timeline_table(tmp_path: Path) -> None:
    _write(
        tmp_path / "data_timeline_treatment.txt",
        "PATIENT_ID\tSTART_DATE\n" "P-001\t0\n",
    )
    out = extract_study_comments(tmp_path)
    assert "timeline_treatment" in out
    assert out["timeline_treatment"].column_comments == {}
