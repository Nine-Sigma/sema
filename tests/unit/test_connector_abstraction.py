"""US-011: orchestration programs against the Connector ABC.

Proves the pipeline accepts any Connector implementation (a non-Databricks
stub here) and that FK sampling reaches the warehouse through the public
`execute_query` method instead of the private `_execute` internal.
"""
import pytest

pytestmark = pytest.mark.unit

from typing import Any
from unittest.mock import MagicMock

from sema.connectors.base import Connector
from sema.models.assertions import Assertion
from sema.models.config import BuildConfig
import sema.pipeline.orchestrate_utils as ou


class StubConnector(Connector):
    """A non-Databricks Connector implementation for tests."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    def extract(
        self, catalog: str, schemas: list[str] | None = None, **kwargs: Any,
    ) -> list[Assertion]:
        return []

    def list_catalogs(self) -> list[str]:
        return ["stub_catalog"]

    def get_datasource_ref(self) -> tuple[str, str, str]:
        return "stub://workspace", "stub", "workspace"

    def execute_query(self, query: str) -> list[tuple[Any, ...]]:
        self.executed.append(query)
        return [(1,)]


class TestConnectorExecuteQuery:
    def test_connector_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            Connector()  # type: ignore[abstract]

    def test_stub_execute_query_needs_no_databricks(self) -> None:
        conn = StubConnector()
        assert conn.execute_query("SELECT 1") == [(1,)]
        assert conn.executed == ["SELECT 1"]


class TestFkDetectionProgramsAgainstConnector:
    def test_fk_sampler_uses_public_execute_query(self, monkeypatch) -> None:
        captured: list[Any] = []

        class CapturingLookup:
            def __init__(self, query_fn: Any, catalog: str, **kwargs: Any) -> None:
                captured.append(query_fn)

        monkeypatch.setattr(ou, "WarehouseSampler", CapturingLookup)
        monkeypatch.setattr(ou, "WarehouseProfileLookup", CapturingLookup)
        monkeypatch.setattr(
            ou, "fetch_columns_by_schema", lambda loader, schema: [],
        )

        connector = StubConnector()
        config = BuildConfig(enable_fk_detection=True, catalog="cat")
        ou.run_fk_detection(
            loader=MagicMock(),
            connector=connector,
            config=config,
            schemas=["s"],
            run_id="run-1",
        )

        assert captured, "sampler/profile lookup never constructed"
        assert all(fn == connector.execute_query for fn in captured)
        assert all(
            getattr(fn, "__name__", "") == "execute_query" for fn in captured
        )
