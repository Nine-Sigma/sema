"""Land a data dictionary given as a PostgreSQL DDL file.

The DDL becomes one relation ``<file stem>__columns`` with one row per column,
so an ``entity_model`` spec binds the same roles over it as over a spreadsheet
dictionary landed by the tabular loader.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

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
from sema.intake.loaders.dictionary_utils import parse_postgres_ddl

LOADER = "dictionary"
LOADER_VERSION = "1"
DICTIONARY_COLUMNS = {
    "entity": "VARCHAR",
    "property": "VARCHAR",
    "raw_type": "VARCHAR",
    "nullable": "BOOLEAN",
    "ordinal": "BIGINT",
    LOCATOR_COLUMN: "VARCHAR",
}


def land_ddl(
    path: Path | str,
    staging: Staging,
    *,
    schema: str,
    policy_class: str = DEFAULT_POLICY_CLASS,
) -> LandResult:
    source = Path(path)
    table = _dictionary_rows(source)
    name = relation_name(f"{source.stem}__columns")
    ensure_schema(staging, schema)
    write_arrow(staging, schema, name, table, DICTIONARY_COLUMNS)
    relation = LandedRelation(
        schema=schema,
        name=name,
        columns=tuple(DICTIONARY_COLUMNS.items()),
        row_count=table.num_rows,
        provenance=provenance(
            source,
            LOADER,
            LOADER_VERSION,
            "row",
            landed_at=utc_now(),
            policy_class=policy_class,
        ),
    )
    return LandResult(
        medium="dictionary",
        source_identity=((source.name, file_digest(source)),),
        relations=(relation,),
        policy_class=policy_class,
    )


def _dictionary_rows(source: Path) -> pa.Table:
    columns: dict[str, list[object]] = {name: [] for name in DICTIONARY_COLUMNS}
    for entity, defs in parse_postgres_ddl(source.read_text()).items():
        for ordinal, column in enumerate(defs, start=1):
            columns["entity"].append(entity)
            columns["property"].append(column.name)
            columns["raw_type"].append(column.postgres_type)
            columns["nullable"].append(column.nullable)
            columns["ordinal"].append(ordinal)
            columns[LOCATOR_COLUMN].append(f"{source.name}!{entity}.{column.name}")
    return pa.table(
        {
            "entity": pa.array(columns["entity"], pa.string()),
            "property": pa.array(columns["property"], pa.string()),
            "raw_type": pa.array(columns["raw_type"], pa.string()),
            "nullable": pa.array(columns["nullable"], pa.bool_()),
            "ordinal": pa.array(columns["ordinal"], pa.int64()),
            LOCATOR_COLUMN: pa.array(columns[LOCATOR_COLUMN], pa.string()),
        }
    )
