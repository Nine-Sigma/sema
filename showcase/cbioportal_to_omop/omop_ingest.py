from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pyarrow as pa

from sema.ingest.duckdb_staging import Staging
from sema.intake.loaders.athena import (
    OPTIONAL_VOCAB_FILES,
    REQUIRED_VOCAB_FILES,
    land_athena_bundle,
)
from showcase.cbioportal_to_omop.omop_ingest_utils import (
    CDM_FIELD_CSV_URL_TEMPLATE,
    CDM_REPO_URL_TEMPLATE,
    load_field_level_comments,
    parse_postgres_ddl,
    postgres_to_duckdb_type,
)
from sema.log import logger

ONTOLOGY_SCHEMA = "ontology_omop"
VOCABULARY_SCHEMA = "vocabulary_omop"

__all__ = [
    "OPTIONAL_VOCAB_FILES",
    "REQUIRED_VOCAB_FILES",
    "fetch_cdm_artifacts",
    "ingest_cdm_schema",
    "ingest_vocabulary",
    "load_field_level_comments",
    "parse_postgres_ddl",
    "postgres_to_duckdb_type",
]


def fetch_cdm_artifacts(version: str = "5.4") -> tuple[str, Path]:
    ddl_url = CDM_REPO_URL_TEMPLATE.format(version=version)
    csv_url = CDM_FIELD_CSV_URL_TEMPLATE.format(version=version)
    logger.info("Fetching OMOP CDM DDL v{} from {}", version, ddl_url)
    ddl = _http_get_text(ddl_url)
    logger.info("Fetching OMOP CDM Field Level CSV v{} from {}", version, csv_url)
    csv_body = _http_get_text(csv_url)
    tmp_csv = Path(tempfile.mkstemp(prefix="omop_fields_", suffix=".csv")[1])
    tmp_csv.write_text(csv_body, encoding="utf-8")
    return ddl, tmp_csv


def _http_get_text(url: str) -> str:
    with urlopen(url) as response:
        data: bytes = response.read()
    return data.decode("utf-8")


def ingest_cdm_schema(
    version: str,
    staging: Staging,
) -> None:
    ddl, fields_csv = fetch_cdm_artifacts(version=version)
    tables = parse_postgres_ddl(ddl)
    comments = load_field_level_comments(fields_csv)
    _stage_cdm_tables(tables, comments, staging)
    logger.info("Staged {} OMOP CDM tables for v{}", len(tables), version)


def _stage_cdm_tables(
    tables: dict[str, list[Any]],
    comments: dict[tuple[str, str], str],
    staging: Staging,
) -> None:
    for table_name, columns in tables.items():
        column_types: dict[str, str] = {}
        column_comments: dict[str, str] = {}
        for col in columns:
            column_types[col.name] = postgres_to_duckdb_type(col.postgres_type)
            comment = comments.get((table_name, col.name.lower()))
            if comment:
                column_comments[col.name] = comment
        empty = _empty_arrow_table(column_types)
        staging.write_table(
            schema=ONTOLOGY_SCHEMA,
            table=table_name,
            rows=empty,
            column_types=column_types,
            column_comments=column_comments,
            table_comment=f"OMOP CDM {table_name}",
        )


def _empty_arrow_table(column_types: dict[str, str]) -> pa.Table:
    arrays: list[pa.Array] = []
    names: list[str] = []
    for name, duckdb_type in column_types.items():
        arrays.append(pa.array([], type=_arrow_type_for(duckdb_type)))
        names.append(name)
    return pa.table(arrays, names=names)


def _arrow_type_for(duckdb_type: str) -> pa.DataType:
    t = duckdb_type.upper()
    if t == "INTEGER":
        return pa.int32()
    if t == "BIGINT":
        return pa.int64()
    if t == "SMALLINT":
        return pa.int16()
    if t == "DOUBLE":
        return pa.float64()
    if t == "BOOLEAN":
        return pa.bool_()
    if t == "DATE":
        return pa.date32()
    if t == "TIMESTAMP":
        return pa.timestamp("us")
    if t == "TIMESTAMPTZ":
        return pa.timestamp("us", tz="UTC")
    return pa.string()


def ingest_vocabulary(
    vocab_path: Path | str | None,
    staging: Staging,
) -> None:
    if vocab_path is None:
        logger.info("No --vocab-path provided; skipping OMOP vocabulary ingestion.")
        return
    land_athena_bundle(Path(vocab_path).expanduser(), staging, schema=VOCABULARY_SCHEMA)
