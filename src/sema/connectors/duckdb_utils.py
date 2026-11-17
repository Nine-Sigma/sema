"""Helpers for the DuckDB source connector (identity, connection, table assembly)."""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from sema.connectors.base import RowCapExceeded, StatementTimeout
from sema.log import logger
from sema.models.assertions import Assertion
from sema.models.config import ProfilingConfig
from sema.models.extraction import ExtractedColumn, ExtractedTable
from sema.pipeline.normalizer import AssertionNormalizer

if TYPE_CHECKING:
    from sema.connectors.databricks import TableWorkItem
    from sema.connectors.duckdb import DuckDBConnector

DUCKDB_PLATFORM = "duckdb"
FETCH_BATCH = 1000

# DuckDB spellings of the types Databricks' connector declines to profile.
TEMPORAL_TYPES = frozenset({
    "DATE", "TIME", "TIMESTAMP", "TIMESTAMP WITH TIME ZONE", "TIMESTAMP_NS",
    "TIMESTAMP_MS", "TIMESTAMP_S", "TIME WITH TIME ZONE", "INTERVAL",
})
NUMERIC_TYPES = frozenset({
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT",
    "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL",
})

_UNSAFE_STEM_CHARS = re.compile(r"[^A-Za-z0-9_-]")


def resolve_database_path(path: str) -> Path:
    """Absolute, user-expanded path; rejects the unset default with a config hint."""
    if not path:
        raise ValueError("DuckDB database path is empty; set DUCKDB_PATH or `duckdb.path`")
    return Path(path).expanduser().resolve()


def duckdb_workspace(path: str) -> str:
    """The file stem for readability and a path digest for uniqueness, with no ``/``."""
    resolved = resolve_database_path(path)
    safe_stem = _UNSAFE_STEM_CHARS.sub("_", resolved.stem)
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:12]
    return f"{safe_stem}-{digest}"


def open_read_only(path: str) -> duckdb.DuckDBPyConnection:
    """Open an existing database read-only; a missing file is a caller error.

    ``read_only=True`` never creates a file, but its ``IOException`` is generic;
    checking first gives the operator the path they mistyped.
    """
    resolved = resolve_database_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"DuckDB database not found: {resolved}")
    return duckdb.connect(str(resolved), read_only=True)


def bounded_read(
    connection: duckdb.DuckDBPyConnection, query: str,
    max_rows: int | None, timeout_s: float | None,
) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Fetch in batches; stop past ``max_rows`` rows, interrupt past ``timeout_s``."""
    deadline = math.inf if timeout_s is None else time.monotonic() + timeout_s
    timer = None if timeout_s is None else threading.Timer(timeout_s, connection.interrupt)
    if timer is not None:
        timer.start()
    rows: list[tuple[Any, ...]] = []
    try:
        cursor = connection.execute(query)
        while batch := cursor.fetchmany(FETCH_BATCH):
            rows.extend(batch)
            if max_rows is not None and len(rows) > max_rows:
                raise RowCapExceeded(f"more than {max_rows} rows")
            if time.monotonic() > deadline:
                raise StatementTimeout(f"past {timeout_s} s")
    except duckdb.InterruptException:
        raise StatementTimeout(f"past {timeout_s} s") from None
    finally:
        if timer is not None:
            timer.cancel()
    return [d[0] for d in cursor.description or []], rows


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def qualified_table(catalog: str, schema: str, table: str) -> str:
    return ".".join(quote_ident(part) for part in (catalog, schema, table))


def should_skip_profiling(data_type: str, profiling: ProfilingConfig) -> bool:
    base_type = data_type.upper().split("(")[0].strip()
    if profiling.skip_temporal_profiling and base_type in TEMPORAL_TYPES:
        return True
    return profiling.skip_numeric_profiling and base_type in NUMERIC_TYPES


def assemble_table_assertions(
    connector: DuckDBConnector,
    work_item: TableWorkItem,
    normalizer: AssertionNormalizer,
    profiling: ProfilingConfig,
) -> list[Assertion]:
    """One table's assertions from the connector's DTO reads."""
    catalog, schema, table = work_item.catalog, work_item.schema, work_item.table_name
    assertions = normalizer.normalize_table(
        ExtractedTable(name=table, catalog=catalog, schema=schema),
    )
    columns = connector.extract_columns_dto(catalog, schema, table)
    for column in columns:
        assertions.extend(normalizer.normalize_column(column))
    for key in connector.extract_foreign_keys_dto(catalog, schema, table):
        assertions.extend(normalizer.normalize_foreign_key(key))
    if profiling.enable_distinct_counts:
        assertions.extend(_profile_columns(connector, work_item, normalizer, profiling, columns))
    try:
        sample = connector.extract_sample_rows_dto(catalog, schema, table, profiling.sample_rows)
    except duckdb.Error as exc:
        logger.warning("Sample rows unreadable for {}: {}", work_item.fqn, exc)
        return assertions
    assertions.extend(normalizer.normalize_sample_rows(sample))
    return assertions


def _profile_columns(
    connector: DuckDBConnector,
    work_item: TableWorkItem,
    normalizer: AssertionNormalizer,
    profiling: ProfilingConfig,
    columns: list[ExtractedColumn],
) -> list[Assertion]:
    assertions: list[Assertion] = []
    for column in columns:
        if should_skip_profiling(column.data_type, profiling):
            continue
        try:
            top_values = connector.extract_top_values_dto(
                work_item.catalog, work_item.schema, work_item.table_name,
                column.name, k=profiling.top_k_values,
            )
        except duckdb.Error as exc:
            logger.warning("Profiling skipped for {}.{}: {}", work_item.fqn, column.name, exc)
            continue
        if top_values is not None:
            assertions.extend(normalizer.normalize_top_values(top_values))
    return assertions
