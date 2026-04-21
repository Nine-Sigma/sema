from __future__ import annotations

from pathlib import Path

DEFAULT_SCHEMAS: tuple[str, ...] = ("cbioportal", "ontology_omop", "vocabulary_omop")


def resolve_db_path(raw: str) -> Path:
    return Path(raw).expanduser()


def escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def qualified(schema: str, table: str) -> str:
    return f"{quote_ident(schema)}.{quote_ident(table)}"


def build_create_table_sql(
    schema: str,
    table: str,
    column_types: dict[str, str],
) -> str:
    cols = ", ".join(
        f"{quote_ident(name)} {col_type}" for name, col_type in column_types.items()
    )
    return f"CREATE TABLE {qualified(schema, table)} ({cols})"


def build_column_comment_sql(
    schema: str, table: str, column: str, comment: str
) -> str:
    return (
        f"COMMENT ON COLUMN {qualified(schema, table)}.{quote_ident(column)} "
        f"IS '{escape_sql_literal(comment)}'"
    )


def build_table_comment_sql(schema: str, table: str, comment: str) -> str:
    return (
        f"COMMENT ON TABLE {qualified(schema, table)} "
        f"IS '{escape_sql_literal(comment)}'"
    )
