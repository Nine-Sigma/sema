"""Lazy warehouse-backed lookups for `JoinDetector`.

`WarehouseSampler` returns bounded distinct-value samples for a column;
`WarehouseProfileLookup` returns `(approx_distinct, row_count)`;
`WarehouseContainment` counts how many of a bounded value set occur in a
whole column. All cache per column (and the profile lookup also caches
`row_count` per table to avoid redundant `COUNT(*)` scans). All fail
closed: a warehouse error returns `None` and is cached, so the detector
downgrades confidence rather than retrying failing queries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sema.connectors.databricks_utils import sql_string_literal
from sema.log import logger

ColumnKey = tuple[str, str, str]
TableKey = tuple[str, str]
QueryFn = Callable[[str], list[tuple[Any, ...]]]


@dataclass
class WarehouseSampler:
    query_fn: QueryFn
    catalog: str
    sample_cap: int = 500
    _cache: dict[ColumnKey, set[str] | None] = field(
        default_factory=dict,
    )

    def __call__(self, key: ColumnKey) -> set[str] | None:
        if key in self._cache:
            return self._cache[key]
        result = self._fetch(key)
        self._cache[key] = result
        return result

    def _fetch(self, key: ColumnKey) -> set[str] | None:
        schema, table, column = key
        sql = (
            f"SELECT DISTINCT `{column}` FROM "
            f"`{self.catalog}`.`{schema}`.`{table}` "
            f"LIMIT {self.sample_cap}"
        )
        try:
            rows = self.query_fn(sql)
        except Exception:
            return None
        return {str(row[0]) for row in rows if row[0] is not None}


@dataclass
class WarehouseProfileLookup:
    query_fn: QueryFn
    catalog: str
    _cache: dict[ColumnKey, tuple[int, int] | None] = field(
        default_factory=dict,
    )
    _row_cache: dict[TableKey, int | None] = field(
        default_factory=dict,
    )

    def __call__(self, key: ColumnKey) -> tuple[int, int] | None:
        if key in self._cache:
            return self._cache[key]
        result = self._fetch(key)
        self._cache[key] = result
        return result

    def _fetch(self, key: ColumnKey) -> tuple[int, int] | None:
        schema, table, column = key
        rows = self._row_count(schema, table)
        if rows is None:
            return None
        try:
            res = self.query_fn(
                f"SELECT COUNT(DISTINCT `{column}`) "
                f"FROM `{self.catalog}`.`{schema}`.`{table}`"
            )
        except Exception:
            return None
        if not res or not res[0]:
            return None
        return (int(res[0][0]), rows)

    def _row_count(self, schema: str, table: str) -> int | None:
        tk = (schema, table)
        if tk in self._row_cache:
            return self._row_cache[tk]
        try:
            res = self.query_fn(
                f"SELECT COUNT(*) FROM "
                f"`{self.catalog}`.`{schema}`.`{table}`"
            )
        except Exception:
            self._row_cache[tk] = None
            return None
        rows = int(res[0][0]) if res and res[0] else None
        self._row_cache[tk] = rows
        return rows


@dataclass
class WarehouseContainment:
    """One statement per call, with exactly the given values as literals."""
    query_fn: QueryFn
    catalog: str
    _cache: dict[tuple[ColumnKey, frozenset[str]], int | None] = field(
        default_factory=dict,
    )

    def __call__(self, key: ColumnKey, values: frozenset[str]) -> int | None:
        cache_key = (key, values)
        if cache_key not in self._cache:
            self._cache[cache_key] = self._fetch(key, values)
        return self._cache[cache_key]

    def _fetch(self, key: ColumnKey, values: frozenset[str]) -> int | None:
        schema, table, column = key
        as_text = f"CAST(`{column}` AS STRING)"
        literals = ", ".join(sql_string_literal(v) for v in sorted(values))
        try:
            res = self.query_fn(
                f"SELECT COUNT(DISTINCT {as_text}) "
                f"FROM `{self.catalog}`.`{schema}`.`{table}` "
                f"WHERE {as_text} IN ({literals})"
            )
        except Exception as exc:
            logger.warning(f"Containment check on {schema}.{table}.{column} failed: {exc}")
            return None
        return int(res[0][0]) if res and res[0] else None


@dataclass
class WarehouseValues:
    """Rows per value of a column, and per (code, text) pair of two columns of one relation.

    Meant for columns of at most the low-cardinality bound; values are read as text, NULL as
    None. A warehouse error returns None and is cached.
    """
    query_fn: QueryFn
    catalog: str
    _cache: dict[tuple[str, ...], dict[Any, int] | None] = field(default_factory=dict)

    def counts(self, key: ColumnKey) -> dict[str | None, int] | None:
        schema, table, column = key
        return self._grouped(schema, table, (column,))

    def pairs(
        self, schema: str, table: str, code: str, text: str,
    ) -> dict[tuple[str | None, str | None], int] | None:
        return self._grouped(schema, table, (code, text))

    def _grouped(self, schema: str, table: str, columns: tuple[str, ...]) -> dict[Any, int] | None:
        cache_key = (schema, table, *columns)
        if cache_key not in self._cache:
            self._cache[cache_key] = self._fetch(schema, table, columns)
        return self._cache[cache_key]

    def _fetch(self, schema: str, table: str, columns: tuple[str, ...]) -> dict[Any, int] | None:
        selected = ", ".join(f"CAST(`{c}` AS STRING)" for c in columns)
        groups = ", ".join(str(i + 1) for i in range(len(columns)))
        try:
            rows = self.query_fn(
                f"SELECT {selected}, COUNT(*) FROM `{self.catalog}`.`{schema}`.`{table}` "
                f"GROUP BY {groups}"
            )
        except Exception as exc:
            logger.warning(f"Value counts on {schema}.{table} {columns} failed: {exc}")
            return None
        if len(columns) == 1:
            return {row[0]: int(row[1]) for row in rows}
        return {tuple(row[:-1]): int(row[-1]) for row in rows}
