"""Helper functions for SQL validation against SCO.

Extracted from validate.py to keep the module focused on the
public validate_sql_against_sco entry point.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot

if TYPE_CHECKING:
    from sema.models.context import SemanticContextObject


def _build_allowed_sets(
    sco: SemanticContextObject,
) -> tuple[set[str], set[str]]:
    allowed_tables: set[str] = set()
    allowed_columns: set[str] = set()

    for asset in sco.physical_assets:
        fqn = f"{asset.catalog}.{asset.schema}.{asset.table}"
        allowed_tables.add(fqn.lower())
        allowed_tables.add(asset.table.lower())
        allowed_tables.add(
            f"{asset.schema}.{asset.table}".lower()
        )

        for col in asset.columns:
            allowed_columns.add(col.lower())

    return allowed_tables, allowed_columns


def _check_references(
    parsed: sqlglot.exp.Expression,
    allowed_tables: set[str],
    allowed_columns: set[str],
) -> list[str]:
    errors: list[str] = []

    for table in parsed.find_all(sqlglot.exp.Table):
        table_name = table.name.lower()
        catalog = table.catalog.lower() if table.catalog else ""
        db = table.db.lower() if table.db else ""

        if catalog and db:
            fqn = f"{catalog}.{db}.{table_name}"
        elif db:
            fqn = f"{db}.{table_name}"
        else:
            fqn = table_name

        if (
            fqn not in allowed_tables
            and table_name not in allowed_tables
        ):
            errors.append(
                f"Unknown table: {table.name}. "
                f"Available: {sorted(allowed_tables)}"
            )

    for column in parsed.find_all(sqlglot.exp.Column):
        col_name = column.name.lower()
        if col_name == "*":
            continue
        if col_name not in allowed_columns:
            errors.append(
                f"Unknown column: {column.name}. "
                f"Available: {sorted(allowed_columns)}"
            )

    return errors
