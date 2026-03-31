"""Unit tests for sema.pipeline.execute.DatabricksExecutor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from sema.models.config import DatabricksConfig
from sema.pipeline.execute import DatabricksExecutor


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


class TestDatabricksExecutor:
    @patch("databricks.sql.connect")
    def test_get_connection_strips_https(self, mock_connect, config):
        mock_connect.return_value = MagicMock()
        executor = DatabricksExecutor(config)
        executor._get_connection()
        mock_connect.assert_called_once_with(
            server_hostname="my-workspace.databricks.com",
            http_path="/sql/1.0/endpoints/abc123",
            access_token="test-token",
        )

    @patch("databricks.sql.connect")
    def test_connection_cached(self, mock_connect, config):
        mock_connect.return_value = MagicMock()
        executor = DatabricksExecutor(config)
        conn1 = executor._get_connection()
        conn2 = executor._get_connection()
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

        executor = DatabricksExecutor(config)
        result = executor.execute("SELECT * FROM users")

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

        executor = DatabricksExecutor(config)
        plan = executor.explain("SELECT * FROM users WHERE id > 5")

        assert "Scan parquet" in plan
        assert "Filter: id > 5" in plan
        cursor.execute.assert_called_with("EXPLAIN SELECT * FROM users WHERE id > 5")

    @patch("databricks.sql.connect")
    def test_close_clears_connection(self, mock_connect, config):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        executor = DatabricksExecutor(config)
        executor._get_connection()
        executor.close()

        mock_conn.close.assert_called_once()
        assert executor._connection is None

    def test_close_noop_when_no_connection(self, config):
        executor = DatabricksExecutor(config)
        executor.close()  # should not raise
