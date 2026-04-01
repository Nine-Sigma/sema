"""Unit tests for sema.runtimes.databricks.DatabricksRuntime."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from sema.models.config import DatabricksConfig
from sema.runtimes.databricks import DatabricksRuntime


@pytest.fixture
def config():
    return DatabricksConfig(
        host="https://my-workspace.databricks.com",
        token="test-token",
        http_path="/sql/1.0/endpoints/abc123",
    )


@pytest.fixture
def mock_connection():
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor


class TestDatabricksRuntime:
    def test_dialect(self, config):
        runtime = DatabricksRuntime(config)
        assert runtime.dialect == "databricks"

    @patch("databricks.sql.connect")
    def test_get_connection_strips_https(self, mock_connect, config):
        mock_connect.return_value = MagicMock()
        runtime = DatabricksRuntime(config)
        runtime._get_connection()
        mock_connect.assert_called_once_with(
            server_hostname="my-workspace.databricks.com",
            http_path="/sql/1.0/endpoints/abc123",
            access_token="test-token",
        )

    @patch("databricks.sql.connect")
    def test_connection_cached(self, mock_connect, config):
        mock_connect.return_value = MagicMock()
        runtime = DatabricksRuntime(config)
        conn1 = runtime._get_connection()
        conn2 = runtime._get_connection()
        assert conn1 is conn2
        mock_connect.assert_called_once()

    @patch("databricks.sql.connect")
    def test_execute_returns_results(self, mock_connect, config):
        cursor = MagicMock()
        cursor.description = [("id",), ("name",)]
        cursor.fetchall.return_value = [(1, "Alice"), (2, "Bob")]
        conn = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = conn

        runtime = DatabricksRuntime(config)
        result = runtime.execute("SELECT * FROM users")

        assert result["columns"] == ["id", "name"]
        assert result["row_count"] == 2
        assert result["rows"] == [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]

    @patch("databricks.sql.connect")
    def test_explain_returns_plan(self, mock_connect, config):
        cursor = MagicMock()
        cursor.fetchall.return_value = [("Scan parquet",), ("Filter: id > 5",)]
        conn = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = conn

        runtime = DatabricksRuntime(config)
        plan = runtime.explain("SELECT * FROM users WHERE id > 5")

        assert "Scan parquet" in plan
        assert "Filter: id > 5" in plan
        cursor.execute.assert_called_with("EXPLAIN SELECT * FROM users WHERE id > 5")

    @patch("databricks.sql.connect")
    def test_close_clears_connection(self, mock_connect, config):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        runtime = DatabricksRuntime(config)
        runtime._get_connection()
        runtime.close()

        mock_conn.close.assert_called_once()
        assert runtime._connection is None

    def test_close_noop_when_no_connection(self, config):
        runtime = DatabricksRuntime(config)
        runtime.close()  # should not raise
