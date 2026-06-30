"""US-013: Databricks connection + source-enumeration helpers (hermetic)."""

from __future__ import annotations

from typing import Any

import pytest

from sema.cli_fit_utils import enumerate_source_databricks, open_databricks_cursor
from sema.models.config import DatabricksConfig

pytestmark = pytest.mark.unit


class _Cursor:
    def __init__(self, distinct: list[Any], count: int) -> None:
        self.statements: list[str] = []
        self._distinct = distinct
        self._count = count
        self._mode = ""

    def execute(self, sql: str, parameters: Any = None) -> "_Cursor":
        self.statements.append(sql)
        self._mode = "distinct" if "DISTINCT" in sql else "count"
        return self

    def fetchall(self) -> list[Any]:
        return self._distinct

    def fetchone(self) -> Any:
        return (self._count,)


def test_enumerate_source_databricks_backticks_and_counts() -> None:
    cur = _Cursor(distinct=[("LUAD",), ("ZZZZ",)], count=42)
    codes, row_count = enumerate_source_databricks(
        cur, schema="study", table="sample", value_column="ONCOTREE_CODE"
    )
    assert codes == ["LUAD", "ZZZZ"]
    assert row_count == 42
    assert "`study`.`sample`" in cur.statements[0]
    assert "`ONCOTREE_CODE`" in cur.statements[0]


def test_open_databricks_cursor_requires_credentials() -> None:
    with pytest.raises(ValueError, match="host / http_path / token"):
        open_databricks_cursor(DatabricksConfig(host="", http_path="", token=""))


def test_open_databricks_cursor_builds_cursor(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _Conn:
        def cursor(self) -> str:
            return "cursor-handle"

    def _fake_connect(**kwargs: Any) -> _Conn:
        captured.update(kwargs)
        return _Conn()

    monkeypatch.setattr("sema.cli_fit_utils.sql_connect", _fake_connect)
    config = DatabricksConfig(
        host="https://example.databricks.com", http_path="/sql/1.0/x", token="tok"
    )
    cursor = open_databricks_cursor(config, catalog="workspace", schema="study")
    assert cursor == "cursor-handle"
    assert captured["server_hostname"] == "example.databricks.com"
    assert captured["http_path"] == "/sql/1.0/x"
    assert captured["access_token"] == "tok"
    assert captured["catalog"] == "workspace"
    assert captured["schema"] == "study"
