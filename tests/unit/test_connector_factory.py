"""One connector choice: `BuildConfig.source` picks the factory for each build worker."""

from __future__ import annotations

import pytest

from sema.connectors.duckdb import DuckDBConnectorFactory
from sema.models.config import BuildConfig
from sema.pipeline.build import DatabricksConnectorFactory, resolve_connector_factory

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("source", "factory"), [
    ("duckdb", DuckDBConnectorFactory),
    ("databricks", DatabricksConnectorFactory),
])
def test_source_picks_its_factory(source: str, factory: type) -> None:
    assert isinstance(resolve_connector_factory(BuildConfig(source=source)), factory)


def test_another_source_is_refused_naming_the_known_ones() -> None:
    with pytest.raises(ValueError, match="databricks") as refused:
        resolve_connector_factory(BuildConfig(source="nosuch"))
    assert "duckdb" in str(refused.value)
