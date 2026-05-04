"""Lazy warehouse-backed lookups for `JoinDetector`.

`WarehouseSampler` returns bounded distinct-value samples for a column;
`WarehouseProfileLookup` returns `(approx_distinct, row_count)`. Both
cache per column (and the profile lookup also caches `row_count` per
table to avoid redundant `COUNT(*)` scans). Both fail closed: a
warehouse error returns `None` and is cached, so the detector
downgrades confidence rather than retrying failing queries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

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
