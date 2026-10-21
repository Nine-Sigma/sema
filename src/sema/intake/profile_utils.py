"""Result shapes and statement builders for the profiling tools.

Every builder takes identifiers already quoted by the relation's provider, so the
statements here never name a relation or column of their own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import takewhile
from typing import Any, Literal

from sema.binding.spec_sections import RelationRef
from sema.ingest.duckdb_staging_utils import escape_sql_literal

Cardinality = Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]

# STRING is the warehouse text type and a DuckDB alias for VARCHAR; the warehouse
# rejects VARCHAR without a length.
TEXT_TYPE = "STRING"
STAGING_DEFAULT_SCHEMA = "main"
AGGREGATES_PER_COLUMN = 3
WHITESPACE_CODE_POINTS = (9, 10, 13, 32)
CARDINALITIES: dict[tuple[bool, bool], Cardinality] = {
    (False, False): "one_to_one",
    (False, True): "one_to_many",
    (True, False): "many_to_one",
    (True, True): "many_to_many",
}


@dataclass(frozen=True)
class CatalogColumn:
    name: str
    type: str
    comment: str | None


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    type: str
    comment: str | None
    null_rate: float
    distinct_count: int
    distinct_ratio: float
    mean_length: float | None
    is_unique: bool
    is_constant: bool


@dataclass(frozen=True)
class RelationProfile:
    relation: str
    row_count: int
    columns: tuple[ColumnProfile, ...]


@dataclass(frozen=True)
class RelationListing:
    name: str
    row_count: int
    comment: str | None


@dataclass(frozen=True)
class JoinCheck:
    resolve_a_to_b: float
    resolve_b_to_a: float
    cardinality: Cardinality


@dataclass(frozen=True)
class HierarchyCheck:
    cycle_count: int
    root_count: int
    max_depth: int


@dataclass(frozen=True)
class LevelCheck:
    misplaced_nulls: int
    multi_parent: int


@dataclass(frozen=True)
class PathCheck:
    code_count: int
    separated: int
    malformed: int
    implicit_count: int
    spaced: int


@dataclass(frozen=True)
class ExternalRefCheck:
    system: str
    resolve_rate: float | None
    unverifiable: bool


@dataclass(frozen=True)
class ValueSample:
    relation: str
    column: str
    values: tuple[Any, ...]


@dataclass(frozen=True)
class NeedsValues:
    """Values the policy withholds; the approval interrupt decides whether to read them."""

    relation: str
    columns: tuple[str, ...]


def sql_string(text: str) -> str:
    return f"'{escape_sql_literal(text)}'"


def schema_and_table(ref: RelationRef) -> tuple[str, str]:
    schema, _, table = ref.name.rpartition(".")
    return schema.rpartition(".")[2] or STAGING_DEFAULT_SCHEMA, table


def staging_columns_sql(ref: RelationRef) -> str:
    schema, table = schema_and_table(ref)
    return (
        "SELECT column_name, data_type, comment FROM duckdb_columns()"
        f" WHERE schema_name = {sql_string(schema)} AND table_name = {sql_string(table)}"
        " ORDER BY column_index"
    )


def staging_tables_sql(namespace: str) -> str:
    return (
        "SELECT table_name, comment FROM duckdb_tables()"
        f" WHERE schema_name = {sql_string(namespace)} ORDER BY table_name"
    )


def warehouse_tables_sql(quoted_catalog: str | None, schema: str) -> str:
    source = f"{quoted_catalog}.information_schema.tables" if quoted_catalog else (
        "information_schema.tables"
    )
    return (
        f"SELECT table_name, comment FROM {source}"
        f" WHERE table_schema = {sql_string(schema)} ORDER BY table_name"
    )


def catalog_columns(rows: Sequence[tuple[Any, ...]]) -> list[CatalogColumn]:
    """Catalog rows as (name, type, comment); a warehouse listing stops at its first section."""
    described = takewhile(lambda row: bool(row[0]) and not str(row[0]).startswith("#"), rows)
    return [
        CatalogColumn(str(row[0]), str(row[1]), _text_or_none(row[2] if len(row) > 2 else None))
        for row in described
    ]


def aggregate_sql(relation_sql: str, quoted_columns: Sequence[str]) -> str:
    parts = ["COUNT(*)"]
    for column in quoted_columns:
        parts += [
            f"COUNT({column})",
            f"COUNT(DISTINCT {column})",
            f"AVG(LENGTH(CAST({column} AS {TEXT_TYPE})))",
        ]
    return f"SELECT {', '.join(parts)} FROM {relation_sql}"


def column_profiles(
    catalog: Sequence[CatalogColumn], row: Sequence[Any]
) -> tuple[int, tuple[ColumnProfile, ...]]:
    rows = int(row[0])
    profiles = []
    for index, column in enumerate(catalog):
        start = 1 + index * AGGREGATES_PER_COLUMN
        nonnull, distinct, mean = row[start : start + AGGREGATES_PER_COLUMN]
        profiles.append(_profile(column, rows, int(nonnull), int(distinct), mean))
    return rows, tuple(profiles)


def _profile(
    column: CatalogColumn, rows: int, nonnull: int, distinct: int, mean: Any
) -> ColumnProfile:
    return ColumnProfile(
        name=column.name,
        type=column.type,
        comment=column.comment,
        null_rate=1.0 - nonnull / rows if rows else 1.0,
        distinct_count=distinct,
        distinct_ratio=distinct / nonnull if nonnull else 0.0,
        mean_length=float(mean) if mean is not None else None,
        is_unique=rows > 0 and distinct == rows,
        is_constant=distinct == 1,
    )


def join_sql(a_sql: str, a_col: str, b_sql: str, b_col: str) -> str:
    def keys(relation: str, column: str) -> str:
        return (
            f"SELECT CAST({column} AS {TEXT_TYPE}) AS k, COUNT(*) AS n FROM {relation}"
            f" WHERE {column} IS NOT NULL GROUP BY 1"
        )

    return (
        f"WITH a AS ({keys(a_sql, a_col)}), b AS ({keys(b_sql, b_col)}) SELECT"
        " (SELECT COUNT(*) FROM a),"
        " (SELECT COUNT(*) FROM a WHERE k IN (SELECT k FROM b)),"
        " (SELECT COUNT(*) FROM b),"
        " (SELECT COUNT(*) FROM b WHERE k IN (SELECT k FROM a)),"
        " (SELECT COALESCE(MAX(n), 0) FROM a WHERE k IN (SELECT k FROM b)),"
        " (SELECT COALESCE(MAX(n), 0) FROM b WHERE k IN (SELECT k FROM a))"
    )


def join_check(row: Sequence[Any]) -> JoinCheck:
    distinct_a, matched_a, distinct_b, matched_b, most_a, most_b = (int(v) for v in row)
    return JoinCheck(
        resolve_a_to_b=matched_a / distinct_a if distinct_a else 0.0,
        resolve_b_to_a=matched_b / distinct_b if distinct_b else 0.0,
        cardinality=CARDINALITIES[most_a > 1, most_b > 1],
    )


def hierarchy_sql(relation_sql: str, id_col: str, parent_col: str) -> str:
    """Cyclic nodes are their own ancestor; depth is walked over the acyclic part only.

    Both recursions use UNION, so each stops once it adds no new row.
    """
    return (
        "WITH RECURSIVE edges AS ("
        f"SELECT DISTINCT CAST({id_col} AS {TEXT_TYPE}) AS child,"
        f" CAST({parent_col} AS {TEXT_TYPE}) AS parent"
        f" FROM {relation_sql} WHERE {id_col} IS NOT NULL),"
        " links AS (SELECT child, parent FROM edges WHERE parent IS NOT NULL),"
        " nodes AS (SELECT child AS node FROM edges UNION SELECT parent FROM links),"
        " ancestry(node, ancestor) AS (SELECT child, parent FROM links UNION"
        " SELECT a.node, l.parent FROM ancestry a JOIN links l ON l.child = a.ancestor),"
        " cyclic AS (SELECT DISTINCT node FROM ancestry WHERE node = ancestor),"
        " roots AS (SELECT node FROM nodes WHERE node NOT IN (SELECT child FROM links)),"
        " walk(node, depth) AS (SELECT node, 0 FROM roots UNION"
        " SELECT l.child, w.depth + 1 FROM walk w JOIN links l ON l.parent = w.node"
        " WHERE l.child NOT IN (SELECT node FROM cyclic))"
        " SELECT (SELECT COUNT(*) FROM cyclic), (SELECT COUNT(*) FROM roots),"
        " (SELECT COALESCE(MAX(depth), 0) FROM walk)"
    )


def distinct_tuples_sql(
    relation_sql: str, quoted_columns: Sequence[str], where: str | None = None
) -> str:
    """Distinct tuples among the rows where every column is non-null."""
    present = " AND ".join(f"{column} IS NOT NULL" for column in quoted_columns)
    scoped = f"({where}) AND {present}" if where else present
    return (
        f"SELECT COUNT(*) FROM (SELECT DISTINCT {', '.join(quoted_columns)}"
        f" FROM {relation_sql} WHERE {scoped}) t"
    )


def count_where_sql(relation_sql: str, conditions: Sequence[str], where: str | None) -> str:
    """The row count, then one count per condition; nothing but counts comes back."""
    counted = [f"COUNT(CASE WHEN {condition} THEN 1 END)" for condition in conditions]
    parts = ["COUNT(*)", *counted]
    scope = f" WHERE {where}" if where else ""
    return f"SELECT {', '.join(parts)} FROM {relation_sql}{scope}"


def levels_sql(relation_sql: str, quoted_columns: Sequence[str]) -> str:
    """Rows with a null above a filled level, and level nodes reached by two prefix paths."""
    misplaced = " OR ".join(
        f"({column} IS NULL AND ({_any_filled(quoted_columns[index + 1 :])}))"
        for index, column in enumerate(quoted_columns[:-1])
    )
    multi = " + ".join(
        f"(SELECT COUNT(*) FROM (SELECT {column} FROM (SELECT DISTINCT"
        f" {', '.join(quoted_columns[: depth + 1])} FROM {relation_sql}"
        f" WHERE {column} IS NOT NULL) p{depth} GROUP BY {column} HAVING COUNT(*) > 1) m{depth})"
        for depth, column in enumerate(quoted_columns)
        if depth > 0
    )
    return f"SELECT (SELECT COUNT(*) FROM {relation_sql} WHERE {misplaced}), {multi}"


def paths_sql(relation_sql: str, column: str, separator: str) -> str:
    """Codes, codes holding the separator, codes with an empty segment, missing prefixes,
    and codes holding whitespace.

    Prefixes are walked by cutting at the last separator, so the separator is one character.
    """
    sep, doubled = sql_string(separator), sql_string(separator * 2)
    spaced = " OR ".join(f"INSTR(code, CHR({point})) > 0" for point in WHITESPACE_CODE_POINTS)

    def parent(code: str) -> str:
        return f"SUBSTR({code}, 1, LENGTH({code}) - INSTR(REVERSE({code}), {sep}))"

    return (
        "WITH RECURSIVE codes AS (SELECT DISTINCT"
        f" CAST({column} AS {TEXT_TYPE}) AS code FROM {relation_sql} WHERE {column} IS NOT NULL),"
        f" prefixes(path) AS (SELECT {parent('code')} FROM codes WHERE INSTR(code, {sep}) > 0"
        f" UNION SELECT {parent('path')} FROM prefixes WHERE INSTR(path, {sep}) > 0)"
        " SELECT (SELECT COUNT(*) FROM codes),"
        f" (SELECT COUNT(*) FROM codes WHERE INSTR(code, {sep}) > 0),"
        f" (SELECT COUNT(*) FROM codes WHERE SUBSTR(code, 1, 1) = {sep}"
        f" OR SUBSTR(code, LENGTH(code), 1) = {sep} OR INSTR(code, {doubled}) > 0),"
        " (SELECT COUNT(*) FROM prefixes WHERE path NOT IN (SELECT code FROM codes)),"
        f" (SELECT COUNT(*) FROM codes WHERE {spaced})"
    )


def distinct_count_sql(relation_sql: str, column: str) -> str:
    return f"SELECT COUNT(DISTINCT {column}) FROM {relation_sql}"


def sample_sql(relation_sql: str, column: str, limit: int) -> str:
    return (
        f"SELECT DISTINCT {column} FROM {relation_sql} WHERE {column} IS NOT NULL"
        f" ORDER BY 1 LIMIT {limit}"
    )


def text_sql(relation_sql: str, quoted_columns: Sequence[str]) -> str:
    return f"SELECT {', '.join(quoted_columns)} FROM {relation_sql}"


def _any_filled(quoted_columns: Sequence[str]) -> str:
    return " OR ".join(f"{column} IS NOT NULL" for column in quoted_columns)


def _text_or_none(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None
