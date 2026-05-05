from __future__ import annotations

from pathlib import Path

from sema.ingest.comment_recovery import ParsedTableComments

from showcase.cbioportal_to_omop.parsers import (
    iter_timeline_files,
    parse_clinical_file,
    timeline_table_name,
)


_FIXED_TABLE_DESCRIPTORS: tuple[tuple[str, str, str], ...] = (
    ("data_sv.txt", "structural_variant",
     "cBioPortal structural variants from data_sv.txt"),
    ("data_cna.txt", "cna",
     "cBioPortal copy-number alterations from data_cna.txt "
     "(pivoted to long format: one row per sample×gene)"),
    ("data_gene_panel_matrix.txt", "gene_panel_matrix",
     "cBioPortal gene panel matrix from data_gene_panel_matrix.txt"),
)

_MAF_FILENAMES: tuple[str, ...] = ("data_mutations.txt", "data_mutations_extended.txt")


def extract_study_comments(study_dir: Path) -> dict[str, ParsedTableComments]:
    out: dict[str, ParsedTableComments] = {}
    _extract_clinical(study_dir, out, "data_clinical_patient.txt", "patient")
    _extract_clinical(study_dir, out, "data_clinical_sample.txt", "sample")
    _extract_supp_clinical(study_dir, out)
    _extract_maf(study_dir, out)
    _extract_fixed_tables(study_dir, out)
    _extract_segmented_cna(study_dir, out)
    _extract_resource_files(study_dir, out)
    _extract_timelines(study_dir, out)
    return out


def _extract_clinical(
    study_dir: Path,
    out: dict[str, ParsedTableComments],
    filename: str,
    table: str,
) -> None:
    path = study_dir / filename
    if not path.exists():
        return
    _, _, column_comments = parse_clinical_file(path)
    out[table] = ParsedTableComments(
        table_comment=f"cBioPortal clinical {table} from {filename}",
        column_comments=column_comments,
    )


def _extract_supp_clinical(
    study_dir: Path, out: dict[str, ParsedTableComments],
) -> None:
    for entry in sorted(study_dir.iterdir()):
        if not (entry.is_file() and entry.name.startswith("data_clinical_supp_")):
            continue
        if not entry.name.endswith(".txt"):
            continue
        table = entry.stem.removeprefix("data_")
        _, _, column_comments = parse_clinical_file(entry)
        out[table] = ParsedTableComments(
            table_comment=f"cBioPortal {table} from {entry.name}",
            column_comments=column_comments,
        )


def _extract_maf(
    study_dir: Path, out: dict[str, ParsedTableComments],
) -> None:
    for candidate in _MAF_FILENAMES:
        path = study_dir / candidate
        if path.exists():
            out["mutation"] = ParsedTableComments(
                table_comment=f"cBioPortal MAF mutations from {candidate}",
                column_comments={},
            )
            return


def _extract_fixed_tables(
    study_dir: Path, out: dict[str, ParsedTableComments],
) -> None:
    for filename, table, comment in _FIXED_TABLE_DESCRIPTORS:
        if not (study_dir / filename).exists():
            continue
        out[table] = ParsedTableComments(
            table_comment=comment, column_comments={},
        )


def _extract_segmented_cna(
    study_dir: Path, out: dict[str, ParsedTableComments],
) -> None:
    seg_files = sorted(study_dir.glob("*.seg"))
    if not seg_files:
        return
    seg = seg_files[0]
    out["cna_segmented"] = ParsedTableComments(
        table_comment=f"cBioPortal segmented CNA from {seg.name}",
        column_comments={},
    )


def _extract_resource_files(
    study_dir: Path, out: dict[str, ParsedTableComments],
) -> None:
    for entry in sorted(study_dir.iterdir()):
        if not (entry.is_file() and entry.name.startswith("data_resource_")):
            continue
        if not entry.name.endswith(".txt"):
            continue
        table = entry.stem.removeprefix("data_")
        out[table] = ParsedTableComments(
            table_comment=f"cBioPortal {table} from {entry.name}",
            column_comments={},
        )


def _extract_timelines(
    study_dir: Path, out: dict[str, ParsedTableComments],
) -> None:
    for kind, path in iter_timeline_files(study_dir):
        table = timeline_table_name(kind)
        out[table] = ParsedTableComments(
            table_comment=f"cBioPortal {table} from {path.name}",
            column_comments={},
        )
