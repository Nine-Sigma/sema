"""5.9 regression guard: connector SQL under the default build config.

Pinned on the pre-slice tree (``8ebb17c``). The default evidence policy is
``full``, so the slice must leave this list byte-identical. Expected to pass
before and after the slice.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from sema.connectors.databricks import DatabricksConnector
from sema.models.config import BuildConfig, DatabricksConfig

pytestmark = pytest.mark.unit

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


def _record_default_extraction() -> list[str]:
    connection = RecordingConnection()
    databricks = DatabricksConfig(
        host="https://fake.example.invalid", http_path="/sql/fake",
        token=SecretStr("fake-token"),
    )
    config = BuildConfig(catalog=CATALOG)
    with patch("sema.connectors.databricks.sql_connect", return_value=connection):
        connector = DatabricksConnector(config=databricks, profiling=config.profiling)
        connector.extract(CATALOG)
    return connection.sql


def _meta(table: str) -> list[str]:
    where = (f"WHERE table_catalog = '{CATALOG}' AND table_schema = '{SCHEMA}' "
             f"AND table_name = '{table}'")
    return [
        "SELECT column_name, data_type, is_nullable, comment "
        f"FROM `system`.`information_schema`.`columns` {where} "
        "ORDER BY ordinal_position",
        "SELECT kcu.constraint_name, kcu.column_name, "
        "ref.table_catalog, ref.table_schema, ref.table_name, ref.column_name "
        "FROM `system`.`information_schema`.`table_constraints` tc "
        "JOIN `system`.`information_schema`.`referential_constraints` rc "
        "ON rc.constraint_catalog = tc.constraint_catalog "
        "AND rc.constraint_schema = tc.constraint_schema "
        "AND rc.constraint_name = tc.constraint_name "
        "JOIN `system`.`information_schema`.`key_column_usage` kcu "
        "ON kcu.constraint_catalog = tc.constraint_catalog "
        "AND kcu.constraint_schema = tc.constraint_schema "
        "AND kcu.constraint_name = tc.constraint_name "
        "JOIN `system`.`information_schema`.`key_column_usage` ref "
        "ON ref.constraint_catalog = rc.unique_constraint_catalog "
        "AND ref.constraint_schema = rc.unique_constraint_schema "
        "AND ref.constraint_name = rc.unique_constraint_name "
        "AND ref.ordinal_position = kcu.position_in_unique_constraint "
        "WHERE tc.constraint_type = 'FOREIGN KEY' "
        f"AND tc.table_catalog = '{CATALOG}' AND tc.table_schema = '{SCHEMA}' "
        f"AND tc.table_name = '{table}' "
        "ORDER BY kcu.constraint_name, kcu.ordinal_position",
        "SELECT column_name, tag_name, tag_value "
        f"FROM `system`.`information_schema`.`column_tags` {where}",
    ]


def _fqn(table: str) -> str:
    return f"`{CATALOG}`.`{SCHEMA}`.`{table}`"


def _distinct(table: str, column: str) -> str:
    return f"SELECT APPROX_COUNT_DISTINCT(`{column}`) FROM {_fqn(table)}"


def _top_k(table: str, column: str) -> str:
    return (f"SELECT `{column}`, COUNT(*) AS freq FROM {_fqn(table)} "
            f"GROUP BY `{column}` ORDER BY freq DESC LIMIT 100")


PINNED_DEFAULT_SQL: list[str] = [
    f"SHOW SCHEMAS IN `{CATALOG}`",
    f"SHOW TABLES IN `{CATALOG}`.`{SCHEMA}`",
    *_meta("patient"),
    _distinct("patient", "patient_id"),
    _distinct("patient", "race_label"),
    _top_k("patient", "race_label"),
    _distinct("patient", "age"),
    _top_k("patient", "age"),
    f"SELECT * FROM {_fqn('patient')} LIMIT 5",
    *_meta("encounter"),
    _distinct("encounter", "encounter_id"),
    f"SELECT * FROM {_fqn('encounter')} LIMIT 5",
]


def test_default_config_connector_sql_matches_pre_slice_recording() -> None:
    """5.9 regression guard: expected green before and after the slice."""
    assert _record_default_extraction() == PINNED_DEFAULT_SQL
