import pytest

pytestmark = pytest.mark.unit

from sema.connectors.base import Connector
from sema.models.assertions import Assertion


class FakeConnector(Connector):
    def extract(self, catalog: str, schemas: list[str] | None = None, **kwargs) -> list[Assertion]:
        return []

    def list_catalogs(self) -> list[str]:
        return ["catalog_a", "catalog_b"]

    def get_datasource_ref(self) -> tuple[str, str, str]:
        return "fake://workspace", "fake", "workspace"


class TestConnectorProtocol:
    def test_extract_returns_assertions(self):
        conn = FakeConnector()
        result = conn.extract(catalog="test")
        assert isinstance(result, list)

    def test_list_catalogs_returns_strings(self):
        conn = FakeConnector()
        result = conn.list_catalogs()
        assert result == ["catalog_a", "catalog_b"]

    def test_connector_is_abstract(self):
        with pytest.raises(TypeError):
            Connector()
