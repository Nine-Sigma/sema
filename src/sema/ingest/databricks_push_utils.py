from __future__ import annotations

import re

COPY_INTO_TABLES: frozenset[tuple[str, str]] = frozenset(
    {
        ("vocabulary_omop", "concept"),
        ("vocabulary_omop", "concept_relationship"),
        ("vocabulary_omop", "concept_ancestor"),
    }
)

COPY_INTO_ROW_THRESHOLD = 100_000

DUCKDB_TO_DATABRICKS_TYPE: dict[str, str] = {
    "INTEGER": "INT",
    "BIGINT": "BIGINT",
    "SMALLINT": "SMALLINT",
    "VARCHAR": "STRING",
    "TEXT": "STRING",
    "DOUBLE": "DOUBLE",
    "FLOAT": "FLOAT",
    "BOOLEAN": "BOOLEAN",
    "DATE": "DATE",
    "TIMESTAMP": "TIMESTAMP_NTZ",
    "TIMESTAMPTZ": "TIMESTAMP",
}


_FORBIDDEN_IDENT_CHARS = (";", "`", "\n", "\r", "\x00")


def back_quote(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def qualified(catalog: str, schema: str, table: str) -> str:
    return f"{back_quote(catalog)}.{back_quote(schema)}.{back_quote(table)}"


def escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def validate_identifier(name: str, kind: str) -> None:
    if not name:
        raise ValueError(f"{kind} identifier must be non-empty")
    for ch in _FORBIDDEN_IDENT_CHARS:
        if ch in name:
            raise ValueError(
                f"{kind} identifier contains forbidden character "
                f"{ch!r}: {name!r}"
            )


def duckdb_to_databricks_type(duckdb_type: str) -> str:
    upper = duckdb_type.strip().upper()
    base = re.sub(r"\s*\([^)]+\)", "", upper)
    return DUCKDB_TO_DATABRICKS_TYPE.get(base, "STRING")


def should_route_via_copy_into(
    schema: str, table: str, row_count: int = 0,
) -> bool:
    if (schema.lower(), table.lower()) in COPY_INTO_TABLES:
        return True
    return row_count >= COPY_INTO_ROW_THRESHOLD


def is_uc_volume_path(uri: str) -> bool:
    return uri.startswith("/Volumes/")


def build_create_schema_sql(catalog: str, schema: str) -> str:
    return f"CREATE SCHEMA IF NOT EXISTS {back_quote(catalog)}.{back_quote(schema)}"


def build_drop_table_sql(catalog: str, schema: str, table: str) -> str:
    return f"DROP TABLE IF EXISTS {qualified(catalog, schema, table)}"


def build_create_table_sql(
    catalog: str,
    schema: str,
    table: str,
    column_specs: list[tuple[str, str, str | None]],
    table_comment: str | None,
) -> str:
    column_clauses = [_column_clause(name, dbx_type, comment) for name, dbx_type, comment in column_specs]
    sql = f"CREATE TABLE {qualified(catalog, schema, table)} (\n  " + ",\n  ".join(column_clauses) + "\n)"
    if table_comment:
        sql += f"\nCOMMENT '{escape_sql_literal(table_comment)}'"
    return sql


def _column_clause(name: str, dbx_type: str, comment: str | None) -> str:
    clause = f"{back_quote(name)} {dbx_type}"
    if comment:
        clause += f" COMMENT '{escape_sql_literal(comment)}'"
    return clause


def build_insert_values_sql(
    catalog: str, schema: str, table: str, column_names: list[str], value_rows: list[list[str]]
) -> str:
    cols = ", ".join(back_quote(n) for n in column_names)
    rows_sql = ", ".join("(" + ", ".join(row) + ")" for row in value_rows)
    return f"INSERT INTO {qualified(catalog, schema, table)} ({cols}) VALUES {rows_sql}"


def format_sql_value(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + escape_sql_literal(str(value)) + "'"


def copy_into_staging_path(staging_uri: str, schema: str, table: str) -> str:
    base = staging_uri.rstrip("/")
    return f"{base}/{schema}/{table}/"


def build_copy_into_sql(catalog: str, schema: str, table: str, staging_uri: str) -> str:
    path = copy_into_staging_path(staging_uri, schema, table)
    return (
        f"COPY INTO {qualified(catalog, schema, table)} "
        f"FROM '{path}' "
        "FILEFORMAT = PARQUET"
    )


def build_count_sql(catalog: str, schema: str, table: str) -> str:
    return f"SELECT COUNT(*) FROM {qualified(catalog, schema, table)}"


def build_alter_column_comment_sql(
    catalog: str, schema: str, table: str, column: str, comment: str,
) -> str:
    validate_identifier(catalog, "catalog")
    validate_identifier(schema, "schema")
    validate_identifier(table, "table")
    validate_identifier(column, "column")
    return (
        f"ALTER TABLE {qualified(catalog, schema, table)} "
        f"ALTER COLUMN {back_quote(column)} "
        f"COMMENT '{escape_sql_literal(comment)}'"
    )


def build_alter_table_comment_sql(
    catalog: str, schema: str, table: str, comment: str,
) -> str:
    validate_identifier(catalog, "catalog")
    validate_identifier(schema, "schema")
    validate_identifier(table, "table")
    return (
        f"COMMENT ON TABLE {qualified(catalog, schema, table)} "
        f"IS '{escape_sql_literal(comment)}'"
    )
