from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator
from urllib.request import Request, urlopen

import pyarrow as pa

from showcase.cbioportal_to_omop.cbioportal_utils import (
    DOWNLOAD_EXACT_FILENAMES,
    DOWNLOAD_PREFIXES,
    EXCLUDED_DOWNLOAD_PREFIXES,
    GITHUB_API_TEMPLATE,
    MEDIA_URL_TEMPLATE,
    RAW_URL_TEMPLATE,
    SKIP_FILENAME_PATTERNS,
    TIMELINE_PATTERN,
    ClinicalHeader,
    cbioportal_type_to_duckdb,
    cna_long_format_rows,
    maf_column_type,
    parse_clinical_header,
    read_clinical_data_rows,
    read_header_block,
    read_tsv_rows,
    rows_to_arrow,
    sv_column_type,
)
from sema.ingest.duckdb_staging import Staging
from sema.log import logger

__all__ = [
    "ClinicalHeader",
    "fetch_study_files",
    "ingest_study",
    "iter_timeline_files",
    "parse_clinical_file",
    "parse_clinical_header",
    "parse_cna_file",
    "parse_gene_panel_matrix",
    "parse_maf",
    "parse_resource_file",
    "parse_sv_file",
    "parse_timeline_file",
]


def fetch_study_files(study_id: str, cache_dir: Path) -> Path:
    study_cache = cache_dir / study_id
    done_marker = study_cache / ".done"
    if done_marker.exists():
        logger.info("Using cached cBioPortal study files at {}", study_cache)
        return study_cache
    study_cache.mkdir(parents=True, exist_ok=True)
    entries = _list_study_entries(study_id)
    downloaded = 0
    for entry in entries:
        if entry.get("type") != "file":
            continue
        name = entry["name"]
        if not _should_download(name):
            continue
        _fetch_lfs_or_raw(study_id, name, study_cache / name)
        downloaded += 1
    done_marker.touch()
    logger.info("Fetched {} cBioPortal files for {}", downloaded, study_id)
    return study_cache


def _fetch_lfs_or_raw(study_id: str, filename: str, target: Path) -> None:
    media_url = MEDIA_URL_TEMPLATE.format(study_id=study_id, filename=filename)
    try:
        _download_url_to(media_url, target)
        return
    except Exception as media_err:
        logger.debug("media URL failed for {}: {}; falling back to raw", filename, media_err)
    raw_url = RAW_URL_TEMPLATE.format(study_id=study_id, filename=filename)
    _download_url_to(raw_url, target)


def _list_study_entries(study_id: str) -> list[dict[str, str]]:
    url = GITHUB_API_TEMPLATE.format(study_id=study_id)
    req = Request(url, headers={"Accept": "application/vnd.github+json"})
    with urlopen(req) as resp:
        data: bytes = resp.read()
    parsed = json.loads(data.decode("utf-8"))
    if not isinstance(parsed, list):
        raise RuntimeError(f"Unexpected GitHub API response for {study_id}: {parsed!r}")
    return parsed


def _should_download(filename: str) -> bool:
    lowered = filename.lower()
    if filename in DOWNLOAD_EXACT_FILENAMES:
        return True
    if any(lowered.startswith(p) for p in EXCLUDED_DOWNLOAD_PREFIXES):
        return False
    return any(filename.startswith(p) for p in DOWNLOAD_PREFIXES)


def _download_url_to(url: str, target: Path) -> None:
    logger.info("Downloading {} -> {}", url, target)
    with urlopen(url) as resp:
        with target.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)


def parse_clinical_file(
    path: Path,
) -> tuple[pa.Table, dict[str, str], dict[str, str]]:
    header_lines = read_header_block(path)
    header = parse_clinical_header(header_lines)
    data_rows, _skipped = read_clinical_data_rows(path, header.column_names)
    column_types = _clinical_column_types(header)
    column_comments = _clinical_column_comments(header)
    table = rows_to_arrow(header.column_names, data_rows, column_types)
    return table, column_types, column_comments


def _clinical_column_types(header: ClinicalHeader) -> dict[str, str]:
    if header.types and len(header.types) == len(header.column_names):
        return {
            name: cbioportal_type_to_duckdb(t)
            for name, t in zip(header.column_names, header.types)
        }
    return {name: "VARCHAR" for name in header.column_names}


def _clinical_column_comments(header: ClinicalHeader) -> dict[str, str]:
    if header.descriptions and len(header.descriptions) == len(header.column_names):
        return {
            name: desc
            for name, desc in zip(header.column_names, header.descriptions)
            if desc
        }
    return {}


def parse_maf(path: Path) -> tuple[pa.Table, dict[str, str], dict[str, str]]:
    column_names, data_rows, _ = read_tsv_rows(path, skip_comment_prefix=True)
    column_types = {name: maf_column_type(name) for name in column_names}
    return rows_to_arrow(column_names, data_rows, column_types), column_types, {}


def parse_timeline_file(
    path: Path,
) -> tuple[pa.Table, dict[str, str], dict[str, str]]:
    column_names, data_rows, _ = read_tsv_rows(path, skip_comment_prefix=True)
    column_types = {name: "VARCHAR" for name in column_names}
    return rows_to_arrow(column_names, data_rows, column_types), column_types, {}


def parse_sv_file(
    path: Path,
) -> tuple[pa.Table, dict[str, str], dict[str, str]]:
    column_names, data_rows, _ = read_tsv_rows(path, skip_comment_prefix=True)
    column_types = {name: sv_column_type(name) for name in column_names}
    return (
        rows_to_arrow(column_names, data_rows, column_types),
        column_types,
        {},
    )


def parse_cna_file(
    path: Path,
) -> tuple[pa.Table, dict[str, str], dict[str, str]]:
    header, data_rows, _ = read_tsv_rows(path, skip_comment_prefix=True)
    long_header, long_rows = cna_long_format_rows(header, data_rows)
    column_types = {
        "sample_id": "VARCHAR",
        "hugo_symbol": "VARCHAR",
        "entrez_gene_id": "BIGINT",
        "cna_value": "INTEGER",
    }
    rows_as_str: list[list[str]] = [
        ["" if v is None else str(v) for v in row]
        for row in long_rows
    ]
    return (
        rows_to_arrow(long_header, rows_as_str, column_types),
        column_types,
        {},
    )


def parse_gene_panel_matrix(
    path: Path,
) -> tuple[pa.Table, dict[str, str], dict[str, str]]:
    column_names, data_rows, _ = read_tsv_rows(path, skip_comment_prefix=True)
    column_types = {name: "VARCHAR" for name in column_names}
    return (
        rows_to_arrow(column_names, data_rows, column_types),
        column_types,
        {},
    )


def parse_resource_file(
    path: Path,
) -> tuple[pa.Table, dict[str, str], dict[str, str]]:
    column_names, data_rows, _ = read_tsv_rows(path, skip_comment_prefix=True)
    column_types = {name: "VARCHAR" for name in column_names}
    return (
        rows_to_arrow(column_names, data_rows, column_types),
        column_types,
        {},
    )


def iter_timeline_files(directory: Path) -> Iterator[tuple[str, Path]]:
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        match = TIMELINE_PATTERN.match(entry.name)
        if match:
            yield match.group("kind"), entry


def _list_skipped_files(directory: Path) -> list[Path]:
    skipped: list[Path] = []
    for entry in sorted(directory.iterdir()):
        if entry.is_dir() and entry.name == "case_lists":
            skipped.append(entry)
            continue
        if not entry.is_file():
            continue
        for pattern in SKIP_FILENAME_PATTERNS:
            if pattern.match(entry.name):
                skipped.append(entry)
                break
    return skipped


def ingest_study(
    study_id: str,
    staging: Staging,
    cache_dir: Path,
) -> None:
    study_dir = fetch_study_files(study_id, cache_dir)
    _ingest_study_dir(study_id, study_dir, staging)


def _ingest_study_dir(study_id: str, study_dir: Path, staging: Staging) -> None:
    for skipped in _list_skipped_files(study_dir):
        logger.info("Skipping unsupported cBioPortal file: {}", skipped.name)

    _try_ingest_clinical(study_dir, staging, "data_clinical_patient.txt", "patient")
    _try_ingest_clinical(study_dir, staging, "data_clinical_sample.txt", "sample")
    _try_ingest_maf(study_dir, staging)
    _try_ingest_fixed_files(study_dir, staging)
    _ingest_prefix_matched_files(
        study_dir, staging,
        prefix="data_resource_", parser=parse_resource_file,
    )
    _ingest_prefix_matched_files(
        study_dir, staging,
        prefix="data_clinical_supp_", parser=parse_clinical_file,
        uses_clinical_comments=True,
    )
    _ingest_timelines(study_dir, staging)
    logger.info("Finished ingesting cBioPortal study {}", study_id)


def _try_ingest_clinical(
    study_dir: Path, staging: Staging, filename: str, table: str
) -> None:
    path = study_dir / filename
    if not path.exists():
        logger.info("cBioPortal file missing, skipping: {}", filename)
        return
    rows, column_types, column_comments = parse_clinical_file(path)
    staging.write_table(
        schema="cbioportal",
        table=table,
        rows=rows,
        column_types=column_types,
        column_comments=column_comments,
        table_comment=f"cBioPortal clinical {table} from {filename}",
    )


_SIMPLE_FIXED_FILES: tuple[tuple[str, str, str], ...] = (
    ("data_sv.txt", "structural_variant",
     "cBioPortal structural variants from data_sv.txt"),
    ("data_cna.txt", "cna",
     "cBioPortal copy-number alterations from data_cna.txt "
     "(pivoted to long format: one row per sample×gene)"),
    ("data_gene_panel_matrix.txt", "gene_panel_matrix",
     "cBioPortal gene panel matrix from data_gene_panel_matrix.txt"),
)

_PARSERS_BY_FILENAME = {
    "data_sv.txt": parse_sv_file,
    "data_cna.txt": parse_cna_file,
    "data_gene_panel_matrix.txt": parse_gene_panel_matrix,
}


def _try_ingest_fixed_files(study_dir: Path, staging: Staging) -> None:
    for filename, table_name, comment in _SIMPLE_FIXED_FILES:
        path = study_dir / filename
        if not path.exists():
            continue
        parser = _PARSERS_BY_FILENAME[filename]
        rows, column_types, _ = parser(path)
        staging.write_table(
            schema="cbioportal",
            table=table_name,
            rows=rows,
            column_types=column_types,
            column_comments={},
            table_comment=comment,
        )


def _ingest_prefix_matched_files(
    study_dir: Path,
    staging: Staging,
    *,
    prefix: str,
    parser: Any,
    uses_clinical_comments: bool = False,
) -> None:
    for entry in sorted(study_dir.iterdir()):
        if not (entry.is_file() and entry.name.startswith(prefix)):
            continue
        if not entry.name.endswith(".txt"):
            continue
        table_name = entry.stem.removeprefix("data_")
        result = parser(entry)
        rows, column_types = result[0], result[1]
        comments = result[2] if uses_clinical_comments else {}
        staging.write_table(
            schema="cbioportal",
            table=table_name,
            rows=rows,
            column_types=column_types,
            column_comments=comments,
            table_comment=f"cBioPortal {table_name} from {entry.name}",
        )


def _try_ingest_maf(study_dir: Path, staging: Staging) -> None:
    for candidate in ("data_mutations.txt", "data_mutations_extended.txt"):
        path = study_dir / candidate
        if path.exists():
            rows, column_types, _ = parse_maf(path)
            staging.write_table(
                schema="cbioportal",
                table="mutation",
                rows=rows,
                column_types=column_types,
                column_comments={},
                table_comment=f"cBioPortal MAF mutations from {candidate}",
            )
            return
    logger.info("cBioPortal MAF missing in {}, skipping", study_dir.name)


def _ingest_timelines(study_dir: Path, staging: Staging) -> None:
    timelines = list(iter_timeline_files(study_dir))
    if not timelines:
        logger.info("No cBioPortal timeline files present in {}", study_dir.name)
        return
    for kind, path in timelines:
        rows, column_types, _ = parse_timeline_file(path)
        staging.write_table(
            schema="cbioportal",
            table=f"timeline_{kind}",
            rows=rows,
            column_types=column_types,
            column_comments={},
            table_comment=f"cBioPortal timeline_{kind} from {path.name}",
        )


