from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterator

import pyarrow as pa

from sema.log import logger

GITHUB_API_TEMPLATE = (
    "https://api.github.com/repos/cBioPortal/datahub/contents/public/{study_id}"
)
MEDIA_URL_TEMPLATE = (
    "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/"
    "{study_id}/{filename}"
)
RAW_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/cBioPortal/datahub/master/public/"
    "{study_id}/{filename}"
)

MAF_NUMERIC_COLUMNS: frozenset[str] = frozenset(
    {
        "Start_Position",
        "End_Position",
        "t_depth",
        "t_ref_count",
        "t_alt_count",
        "n_depth",
        "n_ref_count",
        "n_alt_count",
    }
)

SKIP_FILENAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^data_CNA.*\.txt$"),
    re.compile(r"^data_expression_.*\.txt$"),
    re.compile(r"^data_methylation_.*\.txt$"),
    re.compile(r"^data_linear_CNA.*\.txt$"),
    re.compile(r"^data_log2_CNA.*\.txt$"),
)

TIMELINE_PATTERN = re.compile(r"^data_timeline_(?P<kind>[a-zA-Z0-9_]+)\.txt$")


@dataclass
class ClinicalHeader:
    column_names: list[str] = field(default_factory=list)
    display_names: list[str] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)


def parse_clinical_header(lines: list[str]) -> ClinicalHeader:
    meta_lines: list[list[str]] = []
    data_header_line: list[str] | None = None
    for line in lines:
        if line.startswith("#"):
            meta_lines.append(line.lstrip("#").rstrip("\n").split("\t"))
        else:
            data_header_line = line.rstrip("\n").split("\t")
            break
    header = ClinicalHeader(column_names=data_header_line or [])
    if meta_lines:
        header.display_names = meta_lines[0]
    if len(meta_lines) >= 2:
        header.descriptions = meta_lines[1]
    if len(meta_lines) >= 3:
        header.types = meta_lines[2]
    return header


def cbioportal_type_to_duckdb(type_hint: str) -> str:
    t = type_hint.strip().upper()
    if t == "NUMBER":
        return "DOUBLE"
    if t == "BOOLEAN":
        return "BOOLEAN"
    return "VARCHAR"


def maf_column_type(name: str) -> str:
    if name in MAF_NUMERIC_COLUMNS:
        return "BIGINT"
    return "VARCHAR"


def open_text_defensive(path: Path) -> IO[str]:
    return path.open("r", encoding="utf-8", errors="replace")


def read_tsv_rows(
    path: Path, skip_comment_prefix: bool = True
) -> tuple[list[str], list[list[str]], int]:
    header: list[str] = []
    data_rows: list[list[str]] = []
    skipped = 0
    with open_text_defensive(path) as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if skip_comment_prefix and row and row[0].startswith("#"):
                continue
            if not header:
                header = row
                continue
            if not row or row == [""]:
                continue
            if len(row) != len(header):
                skipped += 1
                logger.warning(
                    "Skipping malformed row in {} (expected {} fields, got {})",
                    path.name,
                    len(header),
                    len(row),
                )
                continue
            data_rows.append(row)
    return header, data_rows, skipped


def read_clinical_data_rows(
    path: Path, expected_columns: list[str]
) -> tuple[list[list[str]], int]:
    data_rows: list[list[str]] = []
    skipped = 0
    expected = len(expected_columns)
    with open_text_defensive(path) as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header_consumed = False
        for row in reader:
            if row and row[0].startswith("#"):
                continue
            if not row or row == [""]:
                continue
            if not header_consumed and row == expected_columns:
                header_consumed = True
                continue
            header_consumed = True
            if len(row) != expected:
                skipped += 1
                logger.warning(
                    "Skipping malformed row in {} (expected {} fields, got {})",
                    path.name,
                    expected,
                    len(row),
                )
                continue
            data_rows.append(row)
    return data_rows, skipped


def rows_to_arrow(
    column_names: list[str],
    data_rows: list[list[str]],
    column_types: dict[str, str],
) -> pa.Table:
    columns: dict[str, list[str | None]] = {name: [] for name in column_names}
    for row in data_rows:
        for name, value in zip(column_names, row):
            columns[name].append(value)
    arrays = [_build_array(columns[name], column_types[name]) for name in column_names]
    return pa.table(arrays, names=column_names)


NULL_PLACEHOLDERS: frozenset[str] = frozenset(
    {"", "NA", "N/A", "NaN", "nan", "null", "NULL", "[Not Available]", "[Unknown]", "[Discrepancy]"}
)


def _is_null_placeholder(value: str | None) -> bool:
    return value is None or value.strip() in NULL_PLACEHOLDERS


def _build_array(values: list[str | None], duckdb_type: str) -> pa.Array:
    t = duckdb_type.upper()
    if t in {"BIGINT", "INTEGER"}:
        ints: list[int | None] = [
            None if _is_null_placeholder(v) else _try_int(v)
            for v in values
        ]
        return pa.array(ints, type=pa.int64())
    if t == "DOUBLE":
        floats: list[float | None] = [
            None if _is_null_placeholder(v) else _try_float(v)
            for v in values
        ]
        return pa.array(floats, type=pa.float64())
    if t == "BOOLEAN":
        bools: list[bool | None] = [
            None if _is_null_placeholder(v)
            else str(v).strip().lower() in {"1", "true", "yes"}
            for v in values
        ]
        return pa.array(bools, type=pa.bool_())
    strings: list[str | None] = [None if v is None else str(v) for v in values]
    return pa.array(strings, type=pa.string())


def _try_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return None


def _try_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def iter_file_lines(path: Path, limit: int | None = None) -> Iterator[str]:
    with open_text_defensive(path) as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            yield line


def read_header_block(path: Path, max_lines: int = 32) -> list[str]:
    lines: list[str] = []
    for line in iter_file_lines(path, limit=max_lines):
        lines.append(line)
        if not line.startswith("#"):
            break
    return lines
