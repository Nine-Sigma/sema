"""US-003: thin SQLGlot-based query layer over the OMOP vocabulary tables.

Hermetic: a fake connection records the rendered SQL + params and returns
canned rows. Asserts that one authored query renders on both DuckDB and
Databricks dialects, that the relationship name and standard flag are passed
in (never hardcoded), and that rows map to the generic ``ConceptRow`` shape.
"""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from showcase.cbioportal_to_omop.omop_policy import OMOP_VOCAB_SCHEMA
from sema.resolve.vocab_store import (
    VocabStore,
    VocabStoreBackend,
    namespace_for_backend,
    open_duckdb_vocab_store,
)
from sema.resolve.vocab_store_utils import (
    ConceptRow,
    VocabStoreSchema,
    concept_by_code_query,
    concept_domain_query,
    maps_to_targets_query,
    row_to_concept,
)

pytestmark = pytest.mark.unit


class _FakeConn:
    """Records the last (sql, params) and returns pre-seeded rows."""

    def __init__(self, rows: list[tuple[Any, ...]] | None = None) -> None:
        self.rows = rows or []
        self.last_sql: str | None = None
        self.last_params: Sequence[Any] | None = None

    def execute(self, sql: str, parameters: Sequence[Any] | None = None) -> "_FakeConn":
        self.last_sql = sql
        self.last_params = parameters
        return self

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


# A small generic schema so the tests never depend on OMOP column literals.
_SCHEMA = VocabStoreSchema(
    concept_table="concept",
    relationship_table="rel",
    synonym_table="syn",
    id_col="id_c",
    code_col="code_c",
    name_col="name_c",
    vocab_col="vocab_c",
    domain_col="domain_c",
    standard_col="std_c",
    invalid_col="inv_c",
    rel_from_col="from_c",
    rel_to_col="to_c",
    rel_id_col="relid_c",
    synonym_name_col="synname_c",
)

_CONCEPT_TUPLE = ("777926", "Lung Adeno", "Condition", "OncoTree", None, "LUAD", None)


def _store(rows: list[tuple[Any, ...]] | None = None, dialect: str = "duckdb") -> tuple[
    VocabStore, _FakeConn
]:
    conn = _FakeConn(rows)
    store = VocabStore(conn, schema=_SCHEMA, namespace="ns", dialect=dialect)
    return store, conn


def test_row_to_concept_maps_positional_tuple() -> None:
    row = row_to_concept(_CONCEPT_TUPLE)
    assert row == ConceptRow(
        id="777926",
        name="Lung Adeno",
        domain="Condition",
        vocabulary="OncoTree",
        standard=None,
        code="LUAD",
        invalid_reason=None,
    )


def test_concept_by_code_renders_namespaced_select_with_two_placeholders() -> None:
    sql = concept_by_code_query(_SCHEMA, "ns").sql(dialect="duckdb")
    assert "FROM ns.concept" in sql
    assert "vocab_c = ?" in sql
    assert "code_c = ?" in sql
    assert sql.count("?") == 2


def test_concept_by_code_returns_typed_row_and_binds_params() -> None:
    store, conn = _store([_CONCEPT_TUPLE])
    row = store.concept_by_code("OncoTree", "LUAD")
    assert row is not None and row.id == "777926" and row.domain == "Condition"
    assert list(conn.last_params or []) == ["OncoTree", "LUAD"]


def test_concept_by_code_returns_none_when_no_rows() -> None:
    store, _ = _store([])
    assert store.concept_by_code("OncoTree", "NOPE") is None


def test_maps_to_targets_joins_and_passes_relationship_from_caller() -> None:
    store, conn = _store([_CONCEPT_TUPLE])
    rows = store.maps_to_targets("777926", relationship_id="Maps to")
    assert len(rows) == 1 and rows[0].vocabulary == "OncoTree"
    sql = conn.last_sql or ""
    assert "JOIN ns.concept" in sql and "ns.rel" in sql
    # relationship value is bound, never embedded as a literal in the SQL.
    assert "Maps to" not in sql
    assert list(conn.last_params or []) == ["777926", "Maps to"]


def test_maps_to_targets_standard_and_validity_filter_pushed_to_sql() -> None:
    store, conn = _store([_CONCEPT_TUPLE])
    store.maps_to_targets(
        "777926", relationship_id="Maps to", standard_flag="S", only_valid=True
    )
    sql = conn.last_sql or ""
    assert "std_c = ?" in sql
    assert "inv_c IS NULL" in sql
    assert list(conn.last_params or []) == ["777926", "Maps to", "S"]


def test_maps_to_targets_no_standard_filter_when_flag_absent() -> None:
    store, conn = _store([])
    store.maps_to_targets("777926", relationship_id="Maps to")
    sql = conn.last_sql or ""
    assert "std_c = ?" not in sql
    assert "IS NULL" not in sql


def test_concept_domain_query_and_value() -> None:
    store, conn = _store([("Condition",)])
    assert store.concept_domain("777926") == "Condition"
    assert "domain_c" in (conn.last_sql or "")
    assert list(conn.last_params or []) == ["777926"]


def test_concept_domain_none_when_absent() -> None:
    store, _ = _store([])
    assert store.concept_domain("000") is None


def test_one_query_renders_on_both_dialects() -> None:
    ast = maps_to_targets_query(_SCHEMA, "workspace.ns", standard=True, only_valid=True)
    duck = ast.sql(dialect="duckdb")
    dbx = ast.sql(dialect="databricks")
    assert "workspace.ns.rel" in duck and "workspace.ns.rel" in dbx
    assert duck.count("?") == dbx.count("?") == 3


def test_no_hand_concatenated_sql_strings_use_ast() -> None:
    # The builders return sqlglot expressions, not str.
    import sqlglot.expressions as exp

    assert isinstance(concept_by_code_query(_SCHEMA, "ns"), exp.Select)
    assert isinstance(maps_to_targets_query(_SCHEMA, "ns"), exp.Select)
    assert isinstance(concept_domain_query(_SCHEMA, "ns"), exp.Select)


def test_omop_schema_carries_real_table_and_column_names() -> None:
    assert OMOP_VOCAB_SCHEMA.concept_table == "concept"
    assert OMOP_VOCAB_SCHEMA.relationship_table == "concept_relationship"
    assert OMOP_VOCAB_SCHEMA.synonym_table == "concept_synonym"


def test_open_duckdb_factory_rejects_missing_file(tmp_path: Any) -> None:
    missing = tmp_path / "nope.duckdb"
    with pytest.raises(FileNotFoundError):
        open_duckdb_vocab_store(str(missing), schema=_SCHEMA)


class _DBAPIConn:
    """A DBAPI-style cursor where execute returns None and fetchall is separate."""

    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows
        self.closed = False

    def execute(self, sql: str, parameters: Sequence[Any] | None = None) -> None:
        return None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows

    def close(self) -> None:
        self.closed = True


def test_dbapi_cursor_path_fetches_via_connection() -> None:
    conn = _DBAPIConn([_CONCEPT_TUPLE])
    store = VocabStore(conn, schema=_SCHEMA, namespace="ns", dialect="databricks")
    row = store.concept_by_code("OncoTree", "LUAD")
    assert row is not None and row.code == "LUAD"


def test_close_delegates_to_connection() -> None:
    conn = _DBAPIConn([])
    store = VocabStore(conn, schema=_SCHEMA, namespace="ns")
    store.close()
    assert conn.closed is True


def test_namespace_for_backend() -> None:
    assert namespace_for_backend(VocabStoreBackend.DUCKDB) == "vocabulary_omop"
    assert (
        namespace_for_backend(VocabStoreBackend.DATABRICKS)
        == "workspace.vocabulary_omop"
    )


def test_concepts_by_ids_preserves_missing_ids() -> None:
    conn = _FakeConn([_CONCEPT_TUPLE])  # only "777926" comes back
    store = VocabStore(conn, schema=_SCHEMA, namespace="ns")
    out = store.concepts_by_ids(["777926", "999"])
    assert set(out) == {"777926", "999"}
    assert out["777926"] is not None and out["777926"].id == "777926"
    assert out["999"] is None  # missing id preserved, not dropped
    assert conn.last_params == ["777926", "999"]


def test_concepts_by_ids_empty_runs_no_query() -> None:
    conn = _FakeConn()
    store = VocabStore(conn, schema=_SCHEMA, namespace="ns")
    assert store.concepts_by_ids([]) == {}
    assert conn.last_sql is None
