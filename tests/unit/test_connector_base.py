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

    def execute_query(self, query: str) -> list[tuple]:
        return []


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


class TestSchemaMetadataDefaults:
    def test_base_read_schema_metadata_is_not_implemented(self):
        with pytest.raises(NotImplementedError):
            FakeConnector().read_schema_metadata("catalog_a", "schema_a")

    def test_base_sql_dialect_is_none(self):
        assert Connector.sql_dialect is None
        assert FakeConnector().sql_dialect is None

    def test_no_new_abstract_methods(self):
        assert Connector.__abstractmethods__ == frozenset(
            {"extract", "list_catalogs", "get_datasource_ref", "execute_query"}
        )
