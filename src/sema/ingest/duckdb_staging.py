from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import duckdb
import pyarrow as pa

from sema.ingest.duckdb_staging_utils import (
    DEFAULT_SCHEMAS,
    build_column_comment_sql,
    build_create_table_sql,
    build_table_comment_sql,
    qualified,
    resolve_db_path,
)


@dataclass
class ColumnInfo:
    type: str
    comment: str | None


@dataclass
class TableInfo:
    columns: dict[str, ColumnInfo] = field(default_factory=dict)
    table_comment: str | None = None


class Staging:
    def __init__(self, db_path: str, schemas: tuple[str, ...] = DEFAULT_SCHEMAS) -> None:
        path = resolve_db_path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._conn = duckdb.connect(str(path))
        self._schemas = schemas
        self._ensure_schemas()

    def _ensure_schemas(self) -> None:
        for schema in self._schemas:
            self._conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')

    def list_schemas(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT schema_name FROM duckdb_schemas()"
        ).fetchall()
        return [r[0] for r in rows]

    def drop_table(self, schema: str, table: str) -> None:
        self._conn.execute(f"DROP TABLE IF EXISTS {qualified(schema, table)}")

    def write_table(
        self,
        schema: str,
        table: str,
        rows: pa.Table | duckdb.DuckDBPyRelation,
        column_types: dict[str, str],
        column_comments: dict[str, str],
        table_comment: str | None,
    ) -> None:
        self.drop_table(schema, table)
        self._conn.execute(build_create_table_sql(schema, table, column_types))
        self._insert_rows(schema, table, rows)
        self._apply_comments(schema, table, column_comments, table_comment)

    def _insert_rows(
        self,
        schema: str,
        table: str,
        rows: pa.Table | duckdb.DuckDBPyRelation,
    ) -> None:
        target = qualified(schema, table)
        if isinstance(rows, pa.Table):
            if rows.num_rows == 0:
                return
            self._conn.register("_arrow_tmp", rows)
            try:
                cols = ", ".join(f'"{c}"' for c in rows.column_names)
                self._conn.execute(
                    f"INSERT INTO {target} ({cols}) SELECT {cols} FROM _arrow_tmp"
                )
            finally:
                self._conn.unregister("_arrow_tmp")
        else:
            self._conn.execute(f"INSERT INTO {target} SELECT * FROM rows", {"rows": rows})

    def _apply_comments(
        self,
        schema: str,
        table: str,
        column_comments: dict[str, str],
        table_comment: str | None,
    ) -> None:
        for column, comment in column_comments.items():
            if comment:
                self._conn.execute(build_column_comment_sql(schema, table, column, comment))
        if table_comment:
            self._conn.execute(build_table_comment_sql(schema, table, table_comment))

    def describe(self, schema: str, table: str) -> TableInfo:
        self._assert_table_exists(schema, table)
        info = TableInfo()
        col_rows = self._conn.execute(
            """
            SELECT column_name, data_type, comment
            FROM duckdb_columns()
            WHERE schema_name = ? AND table_name = ?
            ORDER BY column_index
            """,
            [schema, table],
        ).fetchall()
        for name, dtype, comment in col_rows:
            info.columns[name] = ColumnInfo(type=dtype, comment=comment)
        table_rows = self._conn.execute(
            "SELECT comment FROM duckdb_tables() WHERE schema_name = ? AND table_name = ?",
            [schema, table],
        ).fetchall()
        if table_rows:
            info.table_comment = table_rows[0][0]
        return info

    def _assert_table_exists(self, schema: str, table: str) -> None:
        exists = self._conn.execute(
            "SELECT COUNT(*) FROM duckdb_tables() WHERE schema_name = ? AND table_name = ?",
            [schema, table],
        ).fetchone()
        if not exists or exists[0] == 0:
            raise ValueError(f"Table {schema}.{table} does not exist")

    def execute(self, sql: str, params: Any = None) -> Any:
        return self._conn.execute(sql, params) if params else self._conn.execute(sql)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Staging:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
