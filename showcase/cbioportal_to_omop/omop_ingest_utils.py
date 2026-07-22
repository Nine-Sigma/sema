from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

CDM_REPO_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/OHDSI/CommonDataModel/v{version}/"
    "inst/ddl/{version}/postgresql/OMOPCDM_postgresql_{version}_ddl.sql"
)
CDM_FIELD_CSV_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/OHDSI/CommonDataModel/v{version}/"
    "inst/csv/OMOP_CDMv{version}_Field_Level.csv"
)

CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?:@?[\w.]+\.)?(?P<table>[\w]+)\s*\((?P<body>[^;]+)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)

COLUMN_LINE_RE = re.compile(
    r"^\s*(?P<name>[\w]+)\s+(?P<type>[A-Za-z][\w]*(?:\s+WITH\s+TIME\s+ZONE)?(?:\([^)]+\))?)"
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)


@dataclass
class ColumnDef:
    name: str
    postgres_type: str
    nullable: bool


def postgres_to_duckdb_type(pg_type: str) -> str:
    upper = pg_type.strip().upper()
    upper_no_size = re.sub(r"\s*\([^)]+\)", "", upper)
    if "TIMESTAMP" in upper and "WITH TIME ZONE" in upper:
        return "TIMESTAMPTZ"
    if upper_no_size == "BIGSERIAL":
        return "BIGINT"
    if upper_no_size == "SERIAL":
        return "INTEGER"
    if upper_no_size in {"TEXT", "VARCHAR", "CHAR", "CHARACTER", "CHARACTER VARYING"}:
        return "VARCHAR"
    if upper_no_size in {"INTEGER", "INT", "INT4"}:
        return "INTEGER"
    if upper_no_size in {"BIGINT", "INT8"}:
        return "BIGINT"
    if upper_no_size in {"SMALLINT", "INT2"}:
        return "SMALLINT"
    if upper_no_size in {"NUMERIC", "DECIMAL", "REAL", "DOUBLE", "DOUBLE PRECISION", "FLOAT", "FLOAT4", "FLOAT8"}:
        return "DOUBLE"
    if upper_no_size == "DATE":
        return "DATE"
    if upper_no_size == "TIMESTAMP":
        return "TIMESTAMP"
    if upper_no_size in {"BOOLEAN", "BOOL"}:
        return "BOOLEAN"
    return "VARCHAR"


def parse_postgres_ddl(ddl: str) -> dict[str, list[ColumnDef]]:
    tables: dict[str, list[ColumnDef]] = {}
    for match in CREATE_TABLE_RE.finditer(ddl):
        table_name = match.group("table").lower()
        body = match.group("body")
        columns = _parse_table_body(body)
        if columns:
            tables[table_name] = columns
    return tables


def _parse_table_body(body: str) -> list[ColumnDef]:
    columns: list[ColumnDef] = []
    for raw in _split_top_level_commas(body):
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^\s*(CONSTRAINT|PRIMARY\s+KEY|FOREIGN\s+KEY|UNIQUE|CHECK)\b", line, re.IGNORECASE):
            continue
        m = COLUMN_LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        col_type = m.group("type").strip()
        rest = m.group("rest").upper()
        nullable = "NOT NULL" not in rest
        columns.append(ColumnDef(name=name, postgres_type=col_type, nullable=nullable))
    return columns


def _split_top_level_commas(body: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in body:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def load_field_level_comments(csv_path: Path) -> dict[tuple[str, str], str]:
    comments: dict[tuple[str, str], str] = {}
    with csv_path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            table = (row.get("cdmTableName") or "").strip().lower()
            column = (row.get("cdmFieldName") or "").strip().lower()
            description = (row.get("userGuidance") or "").strip()
            if not table or not column:
                continue
            if description:
                comments[(table, column)] = description
    return comments
