"""Shared setup for test_databricks_sql_pin.py and the tests that reuse it."""

from __future__ import annotations

import re
from typing import Any

CATALOG = "syn_cat"
SCHEMA = "clinical"
TABLES: dict[str, list[tuple[str, str, str, str | None]]] = {
    "patient": [
        ("patient_id", "STRING", "NO", "Primary key"),
        ("race_label", "STRING", "YES", "patient race"),
        ("age", "INT", "YES", None),
        ("birth_date", "DATE", "YES", None),
    ],
    "encounter": [
        ("encounter_id", "STRING", "NO", None),
        ("visit_ts", "TIMESTAMP", "YES", None),
    ],
}
APPROX_DISTINCT = {"patient_id": 9000, "race_label": 5, "age": 90, "encounter_id": 7000}


class RecordingCursor:
    def __init__(self, sql: list[str]) -> None:
        self._sql = sql
        self._rows: list[tuple[Any, ...]] = []
        self.description: list[tuple[str]] | None = None

    def __enter__(self) -> RecordingCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, query: str) -> None:
        self._sql.append(query)
        self._rows, self.description = _respond(query)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


def _respond(query: str) -> tuple[list[tuple[Any, ...]], list[tuple[str]] | None]:
    upper = query.upper()
    if upper.startswith("SHOW SCHEMAS"):
        return [(SCHEMA,)], None
    if upper.startswith("SHOW TABLES"):
        return [(SCHEMA, name, False) for name in TABLES], None
    table = re.search(r"table_name = '(\w+)'", query)
    if "`COLUMNS`" in upper and table:
        return list(TABLES[table.group(1)]), None
    if "APPROX_COUNT_DISTINCT" in upper:
        column = re.search(r"APPROX_COUNT_DISTINCT\(`(\w+)`\)", query)
        return [(APPROX_DISTINCT[column.group(1)] if column else 0,)], None
    if "GROUP BY" in upper:
        return [("v1", 3), ("v2", 1)], None
    if upper.startswith("SELECT *"):
        return [("x",)], [("col",)]
    return [], None


class RecordingConnection:
    def __init__(self) -> None:
        self.sql: list[str] = []

    def cursor(self) -> RecordingCursor:
        return RecordingCursor(self.sql)
