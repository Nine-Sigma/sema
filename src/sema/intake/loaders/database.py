"""Land a DuckDB database file into DuckDB staging, keeping its schema and table names.

The file is attached ``READ_ONLY``, so it is never written and no write-ahead log
appears beside it. Every base table of every schema outside the catalog's internal
ones lands as ``<schema>.<table>`` with its rows, column types, and comments
unchanged; views do not land. Rows land unchanged, so, like the Athena loader, the
loader adds no locator column and records ``locator_kind: relation`` with
``<schema>.<table>`` as the locator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sema.ingest.duckdb_staging import Staging
from sema.ingest.duckdb_staging_utils import (
    build_column_comment_sql,
    build_table_comment_sql,
    escape_sql_literal,
    qualified,
    quote_ident,
)
from sema.intake.landing import LandedRelation, LandResult, utc_now
from sema.intake.loaders.base import (
    DEFAULT_POLICY_CLASS,
    ensure_schema,
    file_digest,
    provenance,
)
from sema.log import logger

LOADER = "database"
LOADER_VERSION = "1"
SOURCE_ALIAS = "_sema_database_source"
INTERNAL_SCHEMAS = ("information_schema", "pg_catalog")


@dataclass(frozen=True)
class _Table:
    schema: str
    name: str
    comment: str | None
    column_comments: tuple[tuple[str, str], ...]


def land_database(
    path: Path | str,
    staging: Staging,
    *,
    schema: str | None = None,
    policy_class: str = DEFAULT_POLICY_CLASS,
) -> LandResult:
    """``schema`` is accepted for the loader call shape; source names are kept as they are."""
    source = Path(path)
    identity = (("sha256", file_digest(source)),)
    staging.execute(
        f"ATTACH '{escape_sql_literal(str(source.resolve()))}' "
        f"AS {quote_ident(SOURCE_ALIAS)} (READ_ONLY)"
    )
    try:
        tables = _base_tables(staging)
        for table in tables:
            _copy(staging, table)
    finally:
        staging.execute(f"DETACH {quote_ident(SOURCE_ALIAS)}")
    landed_at = utc_now()
    relations = [
        _landed(staging, table, source, landed_at, policy_class) for table in tables
    ]
    logger.info("Landed {} relations from database file {}", len(relations), source)
    return LandResult(
        medium="database",
        source_identity=identity,
        relations=tuple(sorted(relations, key=lambda r: (r.schema, r.name))),
        policy_class=policy_class,
    )


def _base_tables(staging: Staging) -> list[_Table]:
    excluded = ", ".join(f"'{name}'" for name in INTERNAL_SCHEMAS)
    rows = staging.execute(
        "SELECT schema_name, table_name, comment FROM duckdb_tables() "
        f"WHERE database_name = ? AND schema_name NOT IN ({excluded}) "
        "ORDER BY schema_name, table_name",
        [SOURCE_ALIAS],
    ).fetchall()
    return [
        _Table(str(schema), str(name), comment, _column_comments(staging, schema, name))
        for schema, name, comment in rows
    ]


def _column_comments(
    staging: Staging, schema: str, name: str
) -> tuple[tuple[str, str], ...]:
    rows = staging.execute(
        "SELECT column_name, comment FROM duckdb_columns() "
        "WHERE database_name = ? AND schema_name = ? AND table_name = ? "
        "AND comment IS NOT NULL AND comment <> '' ORDER BY column_index",
        [SOURCE_ALIAS, schema, name],
    ).fetchall()
    return tuple((str(column), str(comment)) for column, comment in rows)


def _copy(staging: Staging, table: _Table) -> None:
    ensure_schema(staging, table.schema)
    staging.drop_table(table.schema, table.name)
    target = qualified(table.schema, table.name)
    staging.execute(
        f"CREATE TABLE {target} AS SELECT * FROM "
        f"{quote_ident(SOURCE_ALIAS)}.{target}"
    )
    if table.comment:
        staging.execute(build_table_comment_sql(table.schema, table.name, table.comment))
    for column, comment in table.column_comments:
        staging.execute(build_column_comment_sql(table.schema, table.name, column, comment))


def _landed(
    staging: Staging, table: _Table, source: Path, landed_at: str, policy_class: str
) -> LandedRelation:
    info = staging.describe(table.schema, table.name)
    count = staging.execute(
        f"SELECT count(*) FROM {qualified(table.schema, table.name)}"
    ).fetchone()
    locator = f"{table.schema}.{table.name}"
    return LandedRelation(
        schema=table.schema,
        name=table.name,
        columns=tuple((name, column.type) for name, column in info.columns.items()),
        row_count=int(count[0]) if count else 0,
        provenance=provenance(
            source,
            LOADER,
            LOADER_VERSION,
            "relation",
            locator=locator,
            landed_at=landed_at,
            policy_class=policy_class,
        ),
        comments=table.column_comments,
    )
