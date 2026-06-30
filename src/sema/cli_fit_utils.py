"""``sema fit`` Databricks helpers (US-013).

Opening the live connection and enumerating the source on Databricks are the
only pieces the local DuckDB path does not already provide. They live here so
``cli_fit`` stays thin and the live-only code is isolated behind a seam the
hermetic tests can stub.
"""

from __future__ import annotations

from typing import Any, Protocol

from sema.connectors.databricks import sql_connect
from sema.models.config import DatabricksConfig


class _Cursor(Protocol):
    def execute(self, sql: str, parameters: Any = ...) -> Any: ...

    def fetchone(self) -> Any: ...

    def fetchall(self) -> list[Any]: ...


def open_databricks_cursor(
    config: DatabricksConfig, *, catalog: str | None = None, schema: str | None = None
) -> _Cursor:
    """Open a Databricks SQL cursor from the workspace credentials."""
    if not (config.host and config.http_path and config.token.get_secret_value()):
        raise ValueError(
            "Databricks backend needs DATABRICKS host / http_path / token "
            "(set via env or .env)"
        )
    kwargs: dict[str, Any] = {
        "server_hostname": config.host.replace("https://", ""),
        "http_path": config.http_path,
        "access_token": config.token.get_secret_value(),
    }
    if catalog:
        kwargs["catalog"] = catalog
    if schema:
        kwargs["schema"] = schema
    connection = sql_connect(**kwargs)
    return connection.cursor()


def enumerate_source_databricks(
    cursor: _Cursor, *, schema: str, table: str, value_column: str
) -> tuple[list[str], int]:
    """Distinct non-null source codes + total source row count on Databricks."""
    col = f"`{value_column}`"
    tbl = f"`{schema}`.`{table}`"
    cursor.execute(
        f"SELECT DISTINCT {col} FROM {tbl} WHERE {col} IS NOT NULL ORDER BY {col}"
    )
    codes = [str(row[0]) for row in cursor.fetchall()]
    cursor.execute(f"SELECT COUNT(*) FROM {tbl} WHERE {col} IS NOT NULL")
    row = cursor.fetchone()
    return codes, int(row[0]) if row else 0


__all__ = ["enumerate_source_databricks", "open_databricks_cursor"]
