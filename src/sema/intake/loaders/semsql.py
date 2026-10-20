"""Land a prebuilt semantic-sql (semsql) SQLite database into DuckDB staging.

The file is read through the standard library ``sqlite3`` module, read-only,
never through a DuckDB extension (``sqlite_scanner`` is not installed and would
be downloaded on first use). Base tables, the exposed views, and four derived
relations are copied as text columns; every row carries ``_sema_locator``, the
subject IRI where the relation has one and the prefix for ``prefix``.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from sema.ingest.duckdb_staging import Staging
from sema.intake.landing import LOCATOR_COLUMN, LandedRelation, LandResult, utc_now
from sema.intake.loaders.base import (
    DEFAULT_POLICY_CLASS,
    ensure_schema,
    file_digest,
    provenance,
    write_arrow,
)
from sema.intake.loaders.semsql_utils import DERIVED_RELATIONS, ONTOLOGY_SQL, VERSION_SQL
from sema.log import logger

LOADER = "semsql"
LOADER_VERSION = "1"
BASE_TABLES = ("statements", "entailed_edge", "prefix")
EXPOSED_VIEWS = (
    "rdfs_label_statement",
    "has_text_definition_statement",
    "has_exact_synonym_statement",
    "has_dbxref_statement",
    "has_mapping_statement",
    "deprecated_node",
    "class_node",
    "edge",
)
LOCATOR_SOURCES = ("subject", "subject_id", "id", "prefix")


@dataclass(frozen=True)
class _Read:
    name: str
    table: pa.Table


def land_semsql(
    path: Path | str,
    staging: Staging,
    *,
    schema: str,
    policy_class: str = DEFAULT_POLICY_CLASS,
) -> LandResult:
    source = Path(path)
    with closing(sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)) as con:
        reads = _read_all(con, source)
        identity = _identity(con, source)
    ensure_schema(staging, schema)
    landed_at = utc_now()
    relations = [
        _write(staging, schema, read, source, landed_at, policy_class) for read in reads
    ]
    return LandResult(
        medium="semsql",
        source_identity=identity,
        relations=tuple(sorted(relations, key=lambda r: r.name)),
        policy_class=policy_class,
    )


def _read_all(con: sqlite3.Connection, source: Path) -> list[_Read]:
    present = {
        row[0]
        for row in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
    }
    if "statements" not in present:
        raise ValueError(f"{source} is not a semsql database: it has no statements table")
    reads: list[_Read] = []
    for name in BASE_TABLES + EXPOSED_VIEWS:
        if name in present:
            reads.append(_read(con, name, f"SELECT * FROM {name}"))
        else:
            logger.info("Semsql database {} has no {}; not landed", source, name)
    for name, sql in DERIVED_RELATIONS.items():
        try:
            reads.append(_read(con, name, sql))
        except sqlite3.OperationalError as exc:
            raise ValueError(f"{source}: cannot derive {name}: {exc}") from exc
    return reads


def _read(con: sqlite3.Connection, name: str, sql: str) -> _Read:
    cursor = con.execute(sql)
    names = [column[0] for column in cursor.description]
    rows = cursor.fetchall()
    arrays = {
        column: pa.array(
            [None if row[i] is None else str(row[i]) for row in rows], pa.string()
        )
        for i, column in enumerate(names)
    }
    locator = next((column for column in LOCATOR_SOURCES if column in arrays), None)
    arrays[LOCATOR_COLUMN] = (
        arrays[locator] if locator else pa.array([None] * len(rows), pa.string())
    )
    return _Read(name=name, table=pa.table(arrays))


def _identity(con: sqlite3.Connection, source: Path) -> tuple[tuple[str, str], ...]:
    ontology = con.execute(ONTOLOGY_SQL).fetchone()
    if ontology is None:
        return (("sha256", file_digest(source)),)
    version = con.execute(VERSION_SQL, [ontology[0]]).fetchone()
    if version is None or version[0] is None:
        return (("ontology", str(ontology[0])), ("sha256", file_digest(source)))
    return (("ontology", str(ontology[0])), ("version", str(version[0])))


def _write(
    staging: Staging,
    schema: str,
    read: _Read,
    source: Path,
    landed_at: str,
    policy_class: str,
) -> LandedRelation:
    types = {column: "VARCHAR" for column in read.table.column_names}
    write_arrow(staging, schema, read.name, read.table, types)
    return LandedRelation(
        schema=schema,
        name=read.name,
        columns=tuple(types.items()),
        row_count=read.table.num_rows,
        provenance=provenance(
            source, LOADER, LOADER_VERSION, "row", landed_at=landed_at, policy_class=policy_class
        ),
    )
