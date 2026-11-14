"""SQL that reads a CSV as text and casts each column to its declared type, or proves it cannot."""

from __future__ import annotations

import re
from pathlib import Path

from sema.ingest.duckdb_staging import Staging
from sema.ingest.duckdb_staging_utils import escape_sql_literal, quote_ident

from showcase.open_data_load.spec_utils import LoadError, Relation

EXACT_NUMERIC = re.compile(r"^(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|DECIMAL\(.*\))$")
# Wide enough that a value it rounds is beyond every declared exact type's precision.
LOSSLESS_REFERENCE = "DECIMAL(38,18)"
EXAMPLES = 3


def _literal(value: str) -> str:
    return f"'{escape_sql_literal(value)}'"


def position(index: int) -> str:
    return quote_ident(f"c{index}")


def read_csv_sql(path: Path, width: int) -> str:
    columns = ", ".join(f"{_literal(f'c{i}')}: 'VARCHAR'" for i in range(width))
    return (
        f"read_csv({_literal(str(path))}, header = true, auto_detect = false, "
        f"delim = ',', quote = '\"', escape = '\"', columns = {{{columns}}})"
    )


def cast_expr(ref: str, type_: str, date_format: str, *, strict: bool) -> str:
    if type_ == "VARCHAR":
        return ref
    if type_ == "DATE":
        parse = "strptime" if strict else "try_strptime"
        return f"CAST({parse}({ref}, {_literal(date_format)}) AS DATE)"
    return f"{'CAST' if strict else 'TRY_CAST'}({ref} AS {type_})"


def rejects(ref: str, type_: str | None, date_format: str) -> str | None:
    """True for a value the column cannot hold unchanged; None when every value fits."""
    if type_ is None:
        return f"{ref} IS NOT NULL"
    if type_ == "VARCHAR":
        return None
    if type_ == "DATE":
        stamp = f"try_strptime({ref}, {_literal(date_format)})"
        return f"{ref} IS NOT NULL AND ({stamp} IS NULL OR {stamp} <> CAST({stamp} AS DATE))"
    parsed = cast_expr(ref, type_, date_format, strict=False)
    lossy = (
        f" OR TRY_CAST({ref} AS {LOSSLESS_REFERENCE}) IS DISTINCT FROM {parsed}"
        if EXACT_NUMERIC.match(type_) else ""
    )  # fmt: skip
    return f"{ref} IS NOT NULL AND ({parsed} IS NULL{lossy})"


def _checks(
    relation: Relation, headers: list[str], targets: list[str | None], date_format: str
) -> list[tuple[str, str, str]]:
    """(label, column reference, reject predicate) for every column that can reject a value."""
    types = {column.name: column.type for column in relation.columns}
    checks = []
    for index, (header, target) in enumerate(zip(headers, targets, strict=True)):
        type_ = None if target is None else types[target]
        predicate = rejects(position(index), type_, date_format)
        if predicate is not None:
            label = f"{header} (ignored)" if target is None else f"{target} ({type_})"
            checks.append((label, position(index), predicate))
    return checks


def check_casts(
    staging: Staging,
    relation: Relation,
    headers: list[str],
    targets: list[str | None],
    source: str,
    date_format: str,
) -> None:
    """Raise naming each column with a rejected value, its count and three examples."""
    checks = _checks(relation, headers, targets, date_format)
    if not checks:
        return
    counts = staging.execute(
        f"SELECT {', '.join(f'count_if({predicate})' for _, _, predicate in checks)} "
        f"FROM {source}"
    ).fetchone()
    problems = [
        f"{relation.name}: column {label}: {count} values rejected, "
        f"e.g. {_examples(staging, ref, predicate, source)}"
        for (label, ref, predicate), count in zip(checks, counts, strict=True)
        if count
    ]
    if problems:
        raise LoadError("\n".join(problems))


def _examples(staging: Staging, ref: str, predicate: str, source: str) -> str:
    rows = staging.execute(
        f"SELECT {ref} FROM {source} WHERE {predicate} LIMIT {EXAMPLES}"
    ).fetchall()
    return ", ".join(repr(row[0]) for row in rows)


def insert_sql(
    relation: Relation, table: str, targets: list[str | None], source: str, date_format: str
) -> str:
    index = {target: i for i, target in enumerate(targets) if target is not None}
    selects = ", ".join(
        cast_expr(position(index[column.name]), column.type, date_format, strict=True)
        for column in relation.columns
    )
    return f"INSERT INTO {table} SELECT {selects} FROM {source}"
