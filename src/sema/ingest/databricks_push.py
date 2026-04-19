from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from databricks import sql as databricks_sql

from sema.ingest.databricks_push_utils import (
    build_copy_into_sql,
    build_count_sql,
    build_create_schema_sql,
    build_create_table_sql,
    build_drop_table_sql,
    build_insert_values_sql,
    copy_into_staging_path,
    duckdb_to_databricks_type,
    format_sql_value,
    should_route_via_copy_into,
)
from sema.ingest.duckdb_staging import Staging
from sema.log import logger
from sema.models.config import IngestConfig

sql_connect = databricks_sql.connect

INSERT_BATCH_ROWS = 500


@dataclass
class PushResult:
    schema: str
    table: str
    mechanism: str
    rows_pushed: int
    target_count: int
    count_mismatch: bool


class PushError(RuntimeError):
    def __init__(self, failed: list[tuple[str, str, str]]) -> None:
        self.failed = failed
        summary = ", ".join(f"{s}.{t}: {err}" for s, t, err in failed)
        super().__init__(f"Databricks push failed for: {summary}")


class Bridge:
    def __init__(self, config: IngestConfig, staging: Staging) -> None:
        self._config = config
        self._staging = staging
        self._catalog = config.databricks.catalog
        self._schemas = config.databricks.schemas
        self._cloud_uri = config.cloud_staging_uri
        self._connection = self._open_connection()

    def _open_connection(self) -> Any:
        creds = self._config.databricks_creds
        try:
            return sql_connect(
                server_hostname=creds.host.replace("https://", ""),
                http_path=creds.http_path,
                access_token=creds.token.get_secret_value(),
            )
        except Exception as exc:
            raise ConnectionError(f"Failed to connect to Databricks: {exc}") from exc

    def ensure_schemas(self) -> None:
        for schema in self._schemas:
            self._execute(build_create_schema_sql(self._catalog, schema))

    def push_schemas(self, schemas: list[str] | None = None) -> list[PushResult]:
        self.ensure_schemas()
        targets = schemas or self._schemas
        results: list[PushResult] = []
        failures: list[tuple[str, str, str]] = []
        for schema in targets:
            results.extend(self._push_schema_collect(schema, failures))
        if failures:
            raise PushError(failures)
        return results

    def _push_schema_collect(
        self, schema: str, failures: list[tuple[str, str, str]]
    ) -> list[PushResult]:
        results: list[PushResult] = []
        for table in self._list_staged_tables(schema):
            try:
                results.append(self.push_table(schema, table))
            except Exception as exc:
                logger.error("Push failed for {}.{}: {}", schema, table, exc)
                failures.append((schema, table, str(exc)))
        return results

    def push_table(self, schema: str, table: str) -> PushResult:
        self._recreate_target_table(schema, table)
        mechanism, rows_pushed = self._dispatch_push(schema, table)
        target_count = self._count_target(schema, table)
        result = PushResult(
            schema=schema,
            table=table,
            mechanism=mechanism,
            rows_pushed=rows_pushed,
            target_count=target_count,
            count_mismatch=rows_pushed != target_count,
        )
        if result.count_mismatch:
            logger.warning(
                "Row count mismatch on {}.{}: pushed {}, target has {}",
                schema, table, rows_pushed, target_count,
            )
        return result

    def _dispatch_push(self, schema: str, table: str) -> tuple[str, int]:
        if should_route_via_copy_into(schema, table) and self._cloud_uri:
            try:
                return "copy_into", self._push_via_copy_into(schema, table)
            except Exception as exc:
                logger.warning(
                    "COPY INTO failed for {}.{}: {}; falling back to INSERT",
                    schema, table, exc,
                )
        if should_route_via_copy_into(schema, table) and not self._cloud_uri:
            logger.warning(
                "No cloud_staging_uri configured; {}.{} will be loaded via INSERT (slow).",
                schema, table,
            )
        return "insert", self._push_via_insert(schema, table)

    def _recreate_target_table(self, schema: str, table: str) -> None:
        info = self._staging.describe(schema, table)
        self._execute(build_drop_table_sql(self._catalog, schema, table))
        column_specs = [
            (name, duckdb_to_databricks_type(col.type), col.comment)
            for name, col in info.columns.items()
        ]
        self._execute(
            build_create_table_sql(
                self._catalog, schema, table, column_specs, info.table_comment
            )
        )

    def _ddl_from_duckdb(self, schema: str, table: str) -> str:
        info = self._staging.describe(schema, table)
        column_specs = [
            (name, duckdb_to_databricks_type(col.type), col.comment)
            for name, col in info.columns.items()
        ]
        return build_create_table_sql(
            self._catalog, schema, table, column_specs, info.table_comment
        )

    def _push_via_insert(self, schema: str, table: str) -> int:
        info = self._staging.describe(schema, table)
        columns = list(info.columns.keys())
        total = 0
        for batch in self._iter_rows_in_batches(schema, table, columns, INSERT_BATCH_ROWS):
            if not batch:
                continue
            value_rows = [[format_sql_value(v) for v in row] for row in batch]
            self._execute(build_insert_values_sql(self._catalog, schema, table, columns, value_rows))
            total += len(batch)
        return total

    def _iter_rows_in_batches(
        self, schema: str, table: str, columns: list[str], batch_size: int
    ) -> Iterator[list[tuple[object, ...]]]:
        cols = ", ".join(f'"{c}"' for c in columns)
        relation = self._staging.execute(
            f'SELECT {cols} FROM "{schema}"."{table}"'
        )
        while True:
            rows = relation.fetchmany(batch_size)
            if not rows:
                return
            yield rows

    def _push_via_copy_into(self, schema: str, table: str) -> int:
        if not self._cloud_uri:
            raise RuntimeError("copy_into requested without cloud_staging_uri configured")
        rows_pushed = self._export_to_parquet(schema, table, self._cloud_uri)
        self._execute(build_copy_into_sql(self._catalog, schema, table, self._cloud_uri))
        return rows_pushed

    def _export_to_parquet(self, schema: str, table: str, staging_uri: str) -> int:
        target_dir = copy_into_staging_path(staging_uri, schema, table)
        local_dir = _local_path_for_uri(target_dir)
        if local_dir is not None:
            local_dir.mkdir(parents=True, exist_ok=True)
            duckdb_target = str(local_dir / "data.parquet")
        else:
            duckdb_target = target_dir.rstrip("/") + "/data.parquet"
        source = f'"{schema}"."{table}"'
        escaped_target = duckdb_target.replace("'", "''")
        self._staging.execute(
            f"COPY (SELECT * FROM {source}) TO '{escaped_target}' (FORMAT 'parquet')"
        )
        row = self._staging.execute(f"SELECT COUNT(*) FROM {source}").fetchone()
        return int(row[0]) if row else 0

    def _count_target(self, schema: str, table: str) -> int:
        cursor = self._cursor()
        try:
            cursor.execute(build_count_sql(self._catalog, schema, table))
            row = cursor.fetchone()
            return int(row[0]) if row else 0
        finally:
            cursor.close()

    def _list_staged_tables(self, schema: str) -> list[str]:
        rows = self._staging.execute(
            "SELECT table_name FROM duckdb_tables() WHERE schema_name = ? ORDER BY table_name",
            [schema],
        ).fetchall()
        return [r[0] for r in rows]

    def _execute(self, sql: str) -> None:
        cursor = self._cursor()
        try:
            cursor.execute(sql)
        finally:
            cursor.close()

    def _cursor(self) -> Any:
        return self._connection.cursor()

    def close(self) -> None:
        self._connection.close()


def _local_path_for_uri(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        local = parsed.path if parsed.scheme == "file" else uri
        return Path(local)
    return None
