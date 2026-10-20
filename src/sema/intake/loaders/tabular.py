"""Land CSV, XLSX, and Parquet sources as staging relations.

One relation per workbook sheet (``<file stem>__<sheet>``) or per CSV or
Parquet file (``<file stem>``), with non-word characters replaced by ``_``. Every row carries ``_sema_locator`` as
``<sheet>!<physical row>``; CSV and Parquet use the file stem as the sheet.
Merged cells are not expanded: ``openpyxl`` reports only the top-left cell.
"""

from __future__ import annotations

import glob
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq

from sema.ingest.duckdb_staging import Staging
from sema.intake.landing import LOCATOR_COLUMN, LandedRelation, LandResult, utc_now
from sema.intake.loaders.base import (
    DEFAULT_POLICY_CLASS,
    ensure_schema,
    file_digest,
    provenance,
    relation_name,
    write_arrow,
)
from sema.intake.loaders.tabular_utils import (
    HEADER_TESTS,
    cell_text,
    column_names,
    column_type,
    convert,
    detect_header,
)

LOADER = "tabular"
LOADER_VERSION = "1"
SUFFIXES = (".csv", ".parquet", ".xlsx")
ARROW_TYPES = {
    "VARCHAR": pa.string(),
    "BIGINT": pa.int64(),
    "DOUBLE": pa.float64(),
    "BOOLEAN": pa.bool_(),
    "DATE": pa.date32(),
    "TIMESTAMP": pa.timestamp("us"),
}


@dataclass(frozen=True)
class _Sheet:
    path: Path
    label: str
    name: str


@dataclass(frozen=True)
class _SheetData:
    table: pa.Table
    types: dict[str, str]
    header_row: int | None
    empty_columns: tuple[str, ...]


def land_tabular(
    source: Path | str,
    staging: Staging,
    *,
    schema: str,
    policy_class: str = DEFAULT_POLICY_CLASS,
) -> LandResult:
    files = source_files(Path(source))
    sheets = [sheet for path in files for sheet in _plan(path)]
    _check_collisions(sheets)
    read = [(sheet, _read(sheet)) for sheet in sheets]
    ensure_schema(staging, schema)
    landed_at = utc_now()
    relations = [
        _write(staging, schema, sheet, data, landed_at, policy_class)
        for sheet, data in read
    ]
    return LandResult(
        medium="tabular",
        source_identity=tuple((path.name, file_digest(path)) for path in files),
        relations=tuple(sorted(relations, key=lambda r: r.name)),
        policy_class=policy_class,
    )


def source_files(source: Path) -> list[Path]:
    """A file, the supported files of a directory, or the matches of a glob."""
    if source.is_file():
        candidates = [source]
    elif source.is_dir():
        candidates = list(source.iterdir())
    else:
        candidates = [Path(match) for match in glob.glob(str(source))]
    files = sorted(
        (p for p in candidates if p.is_file() and p.suffix.lower() in SUFFIXES),
        key=lambda p: p.name,
    )
    if not files:
        raise FileNotFoundError(f"no {', '.join(SUFFIXES)} file found at {source}")
    return files


def _plan(path: Path) -> list[_Sheet]:
    if path.suffix.lower() != ".xlsx":
        return [_Sheet(path=path, label=path.stem, name=relation_name(path.stem))]
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return [
            _Sheet(path=path, label=title, name=relation_name(f"{path.stem}__{title}"))
            for title in workbook.sheetnames
        ]
    finally:
        workbook.close()


def _check_collisions(sheets: list[_Sheet]) -> None:
    seen: dict[str, _Sheet] = {}
    for sheet in sheets:
        key = sheet.name.lower()
        if key in seen:
            raise ValueError(
                f"relation name {sheet.name} is produced by both "
                f"{seen[key].path.name} and {sheet.path.name}; rename one file"
            )
        seen[key] = sheet


def _read(sheet: _Sheet) -> _SheetData:
    suffix = sheet.path.suffix.lower()
    if suffix == ".parquet":
        return _parquet_data(sheet)
    rows = _xlsx_rows(sheet) if suffix == ".xlsx" else _csv_rows(sheet.path)
    return _rows_data(rows, sheet.label)


def _xlsx_rows(sheet: _Sheet) -> list[list[Any]]:
    workbook = openpyxl.load_workbook(sheet.path, read_only=True, data_only=True)
    try:
        worksheet = workbook[sheet.label]
        return [list(row) for row in worksheet.iter_rows(min_row=1, values_only=True)]
    finally:
        workbook.close()


def _csv_rows(path: Path) -> list[list[Any]]:
    with duckdb.connect(":memory:") as con:
        rows = con.execute(
            "SELECT * FROM read_csv_auto(?, header = false, all_varchar = true)",
            [str(path)],
        ).fetchall()
    return [list(row) for row in rows]


def _rows_data(rows: list[list[Any]], label: str) -> _SheetData:
    header_row = detect_header(rows)
    if header_row is None:
        raise ValueError(
            f"sheet {label}: no row passes the header tests ({'; '.join(HEADER_TESTS)})"
        )
    width = max(len(row) for row in rows)
    names = column_names(rows[header_row - 1], width)
    body = list(_data_rows(rows, header_row, width))
    columns = {name: [row[i] for _, row in body] for i, name in enumerate(names)}
    types = {name: column_type(values) for name, values in columns.items()}
    arrays = {
        name: pa.array([convert(v, types[name]) for v in values], ARROW_TYPES[types[name]])
        for name, values in columns.items()
    }
    arrays[LOCATOR_COLUMN] = pa.array([f"{label}!{number}" for number, _ in body])
    empty = tuple(
        name for name, values in columns.items() if all(cell_text(v) is None for v in values)
    )
    return _SheetData(
        table=pa.table(arrays),
        types={**types, LOCATOR_COLUMN: "VARCHAR"},
        header_row=header_row,
        empty_columns=empty,
    )


def _data_rows(
    rows: list[list[Any]], header_row: int, width: int
) -> Iterator[tuple[int, list[Any]]]:
    for number, row in enumerate(rows[header_row:], start=header_row + 1):
        if any(cell_text(cell) is not None for cell in row):
            yield number, list(row) + [None] * (width - len(row))


def _parquet_data(sheet: _Sheet) -> _SheetData:
    table = pq.read_table(sheet.path)
    with duckdb.connect(":memory:") as con:
        con.register("landed", table)
        described = con.execute("DESCRIBE SELECT * FROM landed").fetchall()
    types = {str(row[0]): str(row[1]) for row in described}
    locators = [f"{sheet.label}!{n}" for n in range(1, table.num_rows + 1)]
    table = table.append_column(LOCATOR_COLUMN, pa.array(locators, pa.string()))
    empty = tuple(
        name for name in table.column_names if table.column(name).null_count == len(table)
    )
    return _SheetData(
        table=table,
        types={**types, LOCATOR_COLUMN: "VARCHAR"},
        header_row=None,
        empty_columns=empty if table.num_rows else (),
    )


def _write(
    staging: Staging,
    schema: str,
    sheet: _Sheet,
    data: _SheetData,
    landed_at: str,
    policy_class: str,
) -> LandedRelation:
    write_arrow(staging, schema, sheet.name, data.table, data.types)
    return LandedRelation(
        schema=schema,
        name=sheet.name,
        columns=tuple(data.types.items()),
        row_count=data.table.num_rows,
        provenance=provenance(
            sheet.path,
            LOADER,
            LOADER_VERSION,
            "row",
            landed_at=landed_at,
            policy_class=policy_class,
        ),
        header_row=data.header_row,
        empty_columns=data.empty_columns,
    )
