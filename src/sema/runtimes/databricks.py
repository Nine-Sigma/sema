"""Databricks SQL runtime implementation."""

from __future__ import annotations

from typing import Any

from sema.log import logger
from sema.models.config import DatabricksConfig


class DatabricksRuntime:
    """Execute SQL against Databricks."""

    def __init__(self, config: DatabricksConfig) -> None:
        self._config = config
        self._connection: Any = None

    @property
    def dialect(self) -> str:
        return "databricks"

    def _get_connection(self) -> Any:
        if self._connection is None:
            from databricks import sql as databricks_sql

            self._connection = databricks_sql.connect(
                server_hostname=self._config.host.replace(
                    "https://", "",
                ),
                http_path=self._config.http_path,
                access_token=self._config.token.get_secret_value(),
            )
        return self._connection

    def execute(self, sql: str) -> dict[str, Any]:
        """Execute SQL and return results."""
        conn = self._get_connection()
        with conn.cursor() as cursor:
            cursor.execute(sql)
            columns = [
                desc[0] for desc in (cursor.description or [])
            ]
            rows = cursor.fetchall()
            results = [dict(zip(columns, row)) for row in rows]
            return {
                "columns": columns,
                "rows": results,
                "row_count": len(results),
            }

    def explain(self, sql: str) -> str:
        """Run EXPLAIN on SQL and return the plan."""
        conn = self._get_connection()
        with conn.cursor() as cursor:
            cursor.execute(f"EXPLAIN {sql}")
            rows = cursor.fetchall()
            return "\n".join(str(row[0]) for row in rows)

    def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None
