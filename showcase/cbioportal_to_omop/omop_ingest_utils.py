from __future__ import annotations

import csv
from pathlib import Path

from sema.intake.land.loaders.dictionary_utils import (
    ColumnDef,
    parse_postgres_ddl,
    postgres_to_duckdb_type,
)

CDM_REPO_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/OHDSI/CommonDataModel/v{version}/"
    "inst/ddl/{version}/postgresql/OMOPCDM_postgresql_{version}_ddl.sql"
)
CDM_FIELD_CSV_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/OHDSI/CommonDataModel/v{version}/"
    "inst/csv/OMOP_CDMv{version}_Field_Level.csv"
)


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


__all__ = [
    "CDM_FIELD_CSV_URL_TEMPLATE",
    "CDM_REPO_URL_TEMPLATE",
    "ColumnDef",
    "load_field_level_comments",
    "parse_postgres_ddl",
    "postgres_to_duckdb_type",
]
