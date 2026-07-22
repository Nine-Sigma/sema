"""bug-368: `delete_table_scoped` clears ONE table's graph footprint so a
resume can re-run a single table without the schema-wide wipe that used to
destroy every sibling table's structural graph.

The primitive anchors on the per-table physical `:Table` / `:Column` nodes
(keyed by name/schema/catalog) and the table's `:Assertion` subject_ref, then
runs the shared orphan sweep. It must NOT emit a schema-wide edge delete
(`{source_schema: $schema}` DELETE r) — that is exactly the blast radius that
wiped sibling tables.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sema.graph.loader import GraphLoader

pytestmark = pytest.mark.unit

CATALOG = "workspace"
SCHEMA = "cbioportal_msk_chord_2024"
TABLE = "sample"
TABLE_REF = "unity://workspace.cbioportal_msk_chord_2024.sample"


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


@pytest.fixture
def loader(mock_driver):
    driver, _ = mock_driver
    return GraphLoader(driver)


def _cyphers(session) -> list[str]:
    return [c[0][0] for c in session.run.call_args_list]


class TestDeleteTableScoped:
    def test_deletes_this_tables_columns(self, loader, mock_driver):
        _, session = mock_driver
        loader.delete_table_scoped(CATALOG, SCHEMA, TABLE, TABLE_REF)
        assert any(
            ":Column" in c
            and "table_name" in c
            and "DETACH DELETE" in c
            for c in _cyphers(session)
        )

    def test_deletes_this_tables_table_node(self, loader, mock_driver):
        _, session = mock_driver
        loader.delete_table_scoped(CATALOG, SCHEMA, TABLE, TABLE_REF)
        assert any(
            ":Table" in c and "DETACH DELETE" in c
            for c in _cyphers(session)
        )

    def test_deletes_table_assertions_by_subject_ref(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        loader.delete_table_scoped(CATALOG, SCHEMA, TABLE, TABLE_REF)
        assert any(
            ":Assertion" in c
            and "subject_ref" in c
            and "STARTS WITH" in c
            and "DETACH DELETE" in c
            for c in _cyphers(session)
        )

    def test_preserve_assertions_skips_assertion_delete(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        loader.delete_table_scoped(
            CATALOG, SCHEMA, TABLE, TABLE_REF, preserve_assertions=True,
        )
        assert not any(
            ":Assertion" in c and "DETACH DELETE a" in c
            for c in _cyphers(session)
        )

    def test_runs_orphan_sweep(self, loader, mock_driver):
        _, session = mock_driver
        loader.delete_table_scoped(CATALOG, SCHEMA, TABLE, TABLE_REF)
        cyphers = _cyphers(session)
        assert any(":Alias" in c and "NOT (a)-[:REFERS_TO]" in c
                   for c in cyphers)
        assert any(":Entity" in c and "NOT (n)--()" in c for c in cyphers)

    def test_never_emits_schema_wide_edge_wipe(self, loader, mock_driver):
        """The whole point of bug-368: table-scoped delete must not do the
        schema-wide `MATCH ()-[r {source_schema}]-() DELETE r` sweep that
        stripped every sibling table's edges."""
        _, session = mock_driver
        loader.delete_table_scoped(CATALOG, SCHEMA, TABLE, TABLE_REF)
        assert not any(
            "source_schema: $schema" in c and "DELETE r" in c
            for c in _cyphers(session)
        )
