"""Physical mapping reads return only the columns IN_TABLE of the returned table."""
from __future__ import annotations

from typing import Any

import pytest

from sema.graph.loader import GraphLoader
from sema.graph.queries import CypherQueries

pytestmark = pytest.mark.integration

CATALOG = "lake"
SCHEMA = "books"
ENTITY = "Entry"
SHARED_PROPERTY = "Identifier"
# Each table's own columns; the shared `id` differs only in data type.
TABLE_COLUMNS: dict[str, dict[str, str]] = {
    "entry_open": {"id": "INT", "amount": "DECIMAL"},
    "entry_closed": {"id": "STRING", "memo": "STRING"},
}
OWN_PROPERTY = {"amount": "Amount", "memo": "Memo"}


@pytest.fixture
def entity_on_two_tables(clean_neo4j: Any) -> Any:
    loader = GraphLoader(clean_neo4j)
    for table, columns in TABLE_COLUMNS.items():
        loader.upsert_table(table, SCHEMA, CATALOG, table_type="TABLE")
        loader.upsert_entity(ENTITY, description=None, source="test", confidence=0.9,
                             table_name=table, schema_name=SCHEMA, catalog=CATALOG,
                             source_schema=SCHEMA)  # fmt: skip
        for column, data_type in columns.items():
            loader.upsert_column(column, table, SCHEMA, CATALOG,
                                 data_type=data_type, nullable=True)  # fmt: skip
            loader.upsert_property(OWN_PROPERTY.get(column, SHARED_PROPERTY),
                                   semantic_type="identifier", source="test",
                                   confidence=0.9, entity_name=ENTITY,
                                   column_name=column, table_name=table,
                                   schema_name=SCHEMA, catalog=CATALOG,
                                   source_schema=SCHEMA)  # fmt: skip
    return clean_neo4j


def _columns_by_table(driver: Any, query: str, **params: Any) -> dict[str, list[tuple[str, str]]]:
    with driver.session() as session:
        rows = [r.data() for r in session.run(query, entity_name=ENTITY, **params)]
    return {
        row["table_name"]: sorted(
            (c["column"], c["data_type"]) for c in row["columns"] if c["column"]
        )
        for row in rows
    }


EXPECTED = {
    table: sorted(columns.items()) for table, columns in TABLE_COLUMNS.items()
}


def test_mapping_returns_only_columns_in_the_returned_table(entity_on_two_tables: Any) -> None:
    got = _columns_by_table(entity_on_two_tables, CypherQueries.resolve_physical_mapping())
    assert got == EXPECTED


def test_scoped_mapping_returns_only_columns_in_the_returned_table(
    entity_on_two_tables: Any,
) -> None:
    got = _columns_by_table(
        entity_on_two_tables, CypherQueries.resolve_physical_mapping_scoped(),
        model_role="SOURCE", schema_name=None,
    )
    assert got == EXPECTED
