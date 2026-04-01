"""SQL validation against SCO-provided assets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot

if TYPE_CHECKING:
    from sema.models.context import SemanticContextObject


def validate_sql_against_sco(
    sql: str,
    sco: SemanticContextObject,
    dialect: str = "databricks",
) -> list[str]:
    """Validate SQL references against SCO physical assets.

    Returns list of error strings. Empty list means valid.
    Checks (table, column) pairs instead of flat sets.
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
    except Exception as e:
        return [f"SQL syntax error: {e}"]

    allowed_tables, allowed_columns = _build_allowed_sets(sco)
    return _check_references(
        parsed, allowed_tables, allowed_columns,  # type: ignore[arg-type]
    )


def _build_allowed_sets(
    sco: SemanticContextObject,
) -> tuple[set[str], set[tuple[str, str]]]:
    """Build allowed table names and (table, column) pairs."""
    allowed_tables: set[str] = set()
    allowed_columns: set[tuple[str, str]] = set()

    for asset in sco.physical_assets:
        fqn = f"{asset.catalog}.{asset.schema}.{asset.table}"
        allowed_tables.add(fqn.lower())
        allowed_tables.add(asset.table.lower())
        allowed_tables.add(
            f"{asset.schema}.{asset.table}".lower()
        )

        for col in asset.columns:
            allowed_columns.add((fqn.lower(), col.lower()))
            allowed_columns.add((asset.table.lower(), col.lower()))

    return allowed_tables, allowed_columns


def _check_references(
    parsed: sqlglot.exp.Expression,
    allowed_tables: set[str],
    allowed_columns: set[tuple[str, str]],
) -> list[str]:
    """Check table and column references against allowed sets."""
    errors: list[str] = []
    _check_table_references(parsed, allowed_tables, errors)
    _check_column_references(
        parsed, allowed_tables, allowed_columns, errors,
    )
    return errors


def _check_table_references(
    parsed: sqlglot.exp.Expression,
    allowed_tables: set[str],
    errors: list[str],
) -> None:
    for table in parsed.find_all(sqlglot.exp.Table):
        table_name = table.name.lower()
        fqn = _build_table_fqn(table)
        if fqn not in allowed_tables and table_name not in allowed_tables:
            closest = _closest_matches(
                table_name, allowed_tables, n=3,
            )
            hint = f" Did you mean: {', '.join(closest)}?" if closest else ""
            errors.append(f"Unknown table: {table.name}.{hint}")


def _check_column_references(
    parsed: sqlglot.exp.Expression,
    allowed_tables: set[str],
    allowed_columns: set[tuple[str, str]],
    errors: list[str],
) -> None:
    all_col_names = {col for _, col in allowed_columns}
    for column in parsed.find_all(sqlglot.exp.Column):
        col_name = column.name.lower()
        if col_name == "*":
            continue
        table_ctx = column.table.lower() if column.table else ""

        if table_ctx:
            if (table_ctx, col_name) not in allowed_columns:
                _add_column_error(
                    column, table_ctx, col_name,
                    allowed_columns, all_col_names, errors,
                )
        elif col_name not in all_col_names:
            closest = _closest_matches(
                col_name, all_col_names, n=3,
            )
            hint = (
                f" Did you mean: {', '.join(closest)}?"
                if closest else ""
            )
            errors.append(f"Unknown column: {column.name}.{hint}")


def _add_column_error(
    column: sqlglot.exp.Column,
    table_ctx: str,
    col_name: str,
    allowed_columns: set[tuple[str, str]],
    all_col_names: set[str],
    errors: list[str],
) -> None:
    """Add a specific error for a column on the wrong table."""
    tables_with_col = [
        t for t, c in allowed_columns if c == col_name
    ]
    if tables_with_col:
        errors.append(
            f"Column {column.name} not found on table "
            f"{column.table}. Found on: "
            f"{', '.join(sorted(set(tables_with_col)))}"
        )
    else:
        closest = _closest_matches(col_name, all_col_names, n=3)
        hint = (
            f" Did you mean: {', '.join(closest)}?"
            if closest else ""
        )
        errors.append(f"Unknown column: {column.name}.{hint}")


def _build_table_fqn(table: sqlglot.exp.Table) -> str:
    """Build a fully qualified table name from AST node."""
    parts: list[str] = []
    if table.catalog:
        parts.append(table.catalog.lower())
    if table.db:
        parts.append(table.db.lower())
    parts.append(table.name.lower())
    return ".".join(parts)


def _closest_matches(
    target: str, candidates: set[str], n: int = 3,
) -> list[str]:
    """Return top-n closest matches by simple edit distance."""
    scored = [
        (c, _simple_distance(target, c)) for c in candidates
    ]
    scored.sort(key=lambda x: x[1])
    return [c for c, _ in scored[:n]]


def _simple_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j in range(1, len(b) + 1):
        curr = [j] + [0] * len(a)
        for i in range(1, len(a) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(
                curr[i - 1] + 1,
                prev[i] + 1,
                prev[i - 1] + cost,
            )
        prev = curr
    return prev[len(a)]
