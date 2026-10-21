"""A fake warehouse connector: schema metadata from DTOs, counted reads, recorded queries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sema.connectors.base import Connector
from sema.models.assertions import Assertion
from sema.models.schema_metadata import (
    ColumnMetadata,
    ObjectMetadata,
    SchemaMetadata,
    Unreadable,
)

CATALOG = "memory"
SCHEMA = "lib"
TABLE = "concept"
UNREADABLE_REASON = "permission denied on schema for the current principal"

CONCEPT_COLUMNS = (
    ColumnMetadata(name="concept_id", data_type="STRING", comment="the concept identifier"),
    ColumnMetadata(name="athena_id", data_type="STRING", comment=None),
    ColumnMetadata(name="vocabulary_id", data_type="STRING", comment="owning vocabulary"),
    ColumnMetadata(name="concept_code", data_type="STRING", comment=None),
    ColumnMetadata(name="definition", data_type="STRING", comment="free-text definition"),
)
CODES_COLUMNS = (
    ColumnMetadata(name="code", data_type="STRING", comment="a code"),
    ColumnMetadata(name="rank", data_type="BIGINT", comment=None),
)


def schema_metadata() -> SchemaMetadata:
    return SchemaMetadata(
        catalog=CATALOG,
        schema_name=SCHEMA,
        objects=(
            ObjectMetadata(
                catalog=CATALOG,
                schema_name=SCHEMA,
                name=TABLE,
                object_type="TABLE",
                columns=CONCEPT_COLUMNS,
                view_definition=None,
            ),
            ObjectMetadata(
                catalog=CATALOG,
                schema_name=SCHEMA,
                name="codes",
                object_type="TABLE",
                columns=CODES_COLUMNS,
                view_definition=None,
            ),
        ),
    )


def unreadable() -> Unreadable:
    return Unreadable(catalog=CATALOG, schema_name=SCHEMA, reason=UNREADABLE_REASON)


class FakeConnector(Connector):
    """Answers ``read_schema_metadata`` from a fixed result and counts every read."""

    sql_dialect = "databricks"

    def __init__(
        self,
        result: SchemaMetadata | Unreadable,
        query: Callable[[str], list[tuple[Any, ...]]] | None = None,
    ) -> None:
        self.result = result
        self.query = query
        self.metadata_reads: list[tuple[str, str]] = []
        self.queries: list[str] = []

    def read_schema_metadata(self, catalog: str, schema: str) -> SchemaMetadata | Unreadable:
        self.metadata_reads.append((catalog, schema))
        return self.result

    def execute_query(self, query: str) -> list[tuple[Any, ...]]:
        self.queries.append(query)
        return self.query(query) if self.query is not None else []

    def extract(
        self, catalog: str, schemas: list[str] | None = None, **kwargs: Any
    ) -> list[Assertion]:
        raise AssertionError("intake Land never runs the legacy extract")

    def list_catalogs(self) -> list[str]:
        return [CATALOG]

    def get_datasource_ref(self) -> tuple[str, str, str]:
        return ("fake://warehouse", "fake", "warehouse")
