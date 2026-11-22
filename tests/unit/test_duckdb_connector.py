"""The DuckDB connector over real temp DuckDB files: read-only, and what the build reads."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest

from sema.connectors.databricks import TableWorkItem
from sema.connectors.duckdb import DuckDBConnector, DuckDBConnectorFactory
from sema.engine.warehouse_lookup import (
    WarehouseContainment,
    WarehouseProfileLookup,
    WarehouseSampler,
)
from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.config import DuckDBConfig, ProfilingConfig
from sema.models.physical_key import CanonicalRef, PhysicalKey

pytestmark = pytest.mark.unit

CATALOG = "fix-cat"
C3_KEYS = {"join_predicates", "from_ref", "to_ref", "constraint_name", "hop_count"}
ODD_COLUMN = ("main", "odd table", "note")
QUOTE_VALUE = "it's"
BACKSLASH_VALUE = "a\\b"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / f"{CATALOG}.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA s1")
    con.execute("CREATE TABLE s1.parent(a INTEGER, b INTEGER, label VARCHAR, PRIMARY KEY(a, b))")
    con.execute(
        "CREATE TABLE s1.child(id INTEGER, y INTEGER, x INTEGER, seen DATE, "
        "FOREIGN KEY (x, y) REFERENCES s1.parent(a, b))"
    )
    con.execute("INSERT INTO s1.parent VALUES (1, 2, 'one'), (3, 4, 'three')")
    con.execute(
        "INSERT INTO s1.child VALUES (10, 2, 1, DATE '2020-01-01'), (11, 2, 1, NULL), "
        "(12, 4, 3, NULL)"
    )
    con.execute('CREATE TABLE main."odd table"(note VARCHAR)')
    con.execute(
        'INSERT INTO main."odd table" VALUES (?), (?), (?), (?), (NULL)',
        [QUOTE_VALUE, BACKSLASH_VALUE, "plain", "plain"],
    )
    con.close()
    return path


def _connector(path: Path, profiling: ProfilingConfig | None = None) -> DuckDBConnector:
    return DuckDBConnector(DuckDBConfig(path=str(path)), profiling or ProfilingConfig())


@pytest.fixture
def connector(db_path: Path) -> Iterator[DuckDBConnector]:
    conn = _connector(db_path)
    yield conn
    conn.close()


def _item(connector: DuckDBConnector, table: str) -> TableWorkItem:
    return next(i for i in connector.discover_tables(CATALOG, ["s1"]) if i.table_name == table)


def _of(assertions: list[Assertion], predicate: AssertionPredicate) -> list[Assertion]:
    return [a for a in assertions if a.predicate == predicate]


class TestReadOnly:
    @pytest.mark.parametrize("statement", [
        "CREATE TABLE s1.extra (k INTEGER)",
        "INSERT INTO s1.parent VALUES (9, 9, 'nine')",
    ])
    def test_writes_are_refused(self, connector: DuckDBConnector, statement: str) -> None:
        with pytest.raises(duckdb.Error):
            connector.execute_query(statement)

    def test_missing_file_raises_and_is_not_created(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.duckdb"
        with pytest.raises(FileNotFoundError):
            _connector(missing)
        assert not missing.exists()


class TestDiscovery:
    def test_catalog_is_the_file_name(self, connector: DuckDBConnector) -> None:
        assert connector.list_catalogs() == [CATALOG]

    def test_discover_tables_returns_work_items_with_parsable_refs(
        self, connector: DuckDBConnector,
    ) -> None:
        ref, platform, workspace = connector.get_datasource_ref()
        items = connector.discover_tables(CATALOG, ["s1"])
        assert all(isinstance(item, TableWorkItem) for item in items)
        assert {item.table_name for item in items} == {"parent", "child"}
        assert (platform, ref) == ("duckdb", f"duckdb://{workspace}")
        for item in items:
            assert CanonicalRef.parse(item.fqn) == PhysicalKey(
                workspace, CATALOG, "s1", item.table_name,
            )

    def test_discover_tables_with_no_schemas_includes_the_default_schema(
        self, connector: DuckDBConnector,
    ) -> None:
        found = {(item.schema, item.table_name) for item in connector.discover_tables(CATALOG)}
        assert found == {("s1", "parent"), ("s1", "child"), ("main", "odd table")}
        assert not {schema for schema, _ in found} & {"information_schema", "pg_catalog"}

    def test_factory_opens_one_connection_per_connector(self, db_path: Path) -> None:
        factory = DuckDBConnectorFactory(DuckDBConfig(path=str(db_path)), ProfilingConfig())
        first, second = factory.create(), factory.create()
        assert first is not second
        first.close()
        count = f"SELECT COUNT(*) FROM `{CATALOG}`.`s1`.`parent`"
        assert second.execute_query(count) == [(2,)]
        second.close()


class TestTranslatedLookups:
    """The build's Databricks-dialect lookups, over a hyphenated catalog and a quoted table."""

    def test_sampler_returns_the_distinct_values(self, connector: DuckDBConnector) -> None:
        sampler = WarehouseSampler(query_fn=connector.execute_query, catalog=CATALOG)
        assert sampler(ODD_COLUMN) == {QUOTE_VALUE, BACKSLASH_VALUE, "plain"}

    def test_profile_returns_distinct_and_row_counts(self, connector: DuckDBConnector) -> None:
        lookup = WarehouseProfileLookup(query_fn=connector.execute_query, catalog=CATALOG)
        assert lookup(ODD_COLUMN) == (3, 5)

    def test_containment_counts_values_holding_a_quote_and_a_backslash(
        self, connector: DuckDBConnector,
    ) -> None:
        containment = WarehouseContainment(query_fn=connector.execute_query, catalog=CATALOG)
        values = frozenset({QUOTE_VALUE, BACKSLASH_VALUE, "absent"})
        assert containment(ODD_COLUMN, values) == 2


class TestExtractTable:
    def test_sample_rows_are_dicts_keyed_by_column(self, connector: DuckDBConnector) -> None:
        [sample] = _of(
            connector.extract_table(_item(connector, "child")),
            AssertionPredicate.HAS_SAMPLE_ROWS,
        )
        rows = sample.payload["rows"]
        assert rows and all(isinstance(row, dict) for row in rows)
        assert all(list(row) == ["id", "y", "x", "seen"] for row in rows)

    @pytest.mark.parametrize("enabled", [True, False])
    def test_distinct_counts_follow_the_profiling_flag(self, db_path: Path, enabled: bool) -> None:
        conn = _connector(db_path, ProfilingConfig(enable_distinct_counts=enabled))
        try:
            top_values = _of(
                conn.extract_table(_item(conn, "parent")), AssertionPredicate.HAS_TOP_VALUES,
            )
        finally:
            conn.close()
        assert bool(top_values) is enabled

    def test_declared_key_is_one_c3_join_in_ordinal_order(
        self, connector: DuckDBConnector,
    ) -> None:
        child, parent = _item(connector, "child"), _item(connector, "parent")
        [join] = _of(connector.extract_table(child), AssertionPredicate.JOINS_TO)
        assert set(join.payload) == C3_KEYS
        assert (join.subject_ref, join.object_ref) == (child.fqn, parent.fqn)
        assert (join.payload["to_ref"], join.payload["from_ref"]) == (child.fqn, parent.fqn)
        assert join.payload["hop_count"] == 1
        assert join.payload["constraint_name"]
        assert [
            (p["left_table"], p["left_column"], p["right_table"], p["right_column"], p["operator"])
            for p in join.payload["join_predicates"]
        ] == [("parent", "a", "child", "x", "="), ("parent", "b", "child", "y", "=")]

    def test_a_failed_profiling_statement_skips_only_its_column(
        self, connector: DuckDBConnector, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        profile_column = connector.extract_top_values_dto

        def fail_on_label(catalog: str, schema: str, table: str, column: str, k: int = 20) -> Any:
            if column == "label":
                raise duckdb.ConversionException("unreadable column")
            return profile_column(catalog, schema, table, column, k)

        monkeypatch.setattr(connector, "extract_top_values_dto", fail_on_label)
        assertions = connector.extract_table(_item(connector, "parent"))
        profiled = {
            CanonicalRef.parse(a.subject_ref).column
            for a in _of(assertions, AssertionPredicate.HAS_TOP_VALUES)
        }
        assert profiled == {"a", "b"}
        assert _of(assertions, AssertionPredicate.HAS_SAMPLE_ROWS)


def test_a_connector_without_external_access_reads_its_tables_and_no_local_file(
    db_path: Path, tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    connector = DuckDBConnector(DuckDBConfig(path=str(db_path), external_access=False))
    try:
        assert connector.execute_query_with_columns("SELECT count(*) AS n FROM s1.parent")[1] == [(2,)]
        for statement in (
            f"SELECT content FROM read_text('{outside}')",
            f"COPY (SELECT 1) TO '{tmp_path / 'written.csv'}'",
            "SET enable_external_access = true",
        ):
            with pytest.raises(duckdb.Error):
                connector.execute_query_with_columns(statement)
    finally:
        connector.close()
    assert not (tmp_path / "written.csv").exists()
