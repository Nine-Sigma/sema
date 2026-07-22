"""S1-06 — FK-closed multi-table compiler (hermetic DuckDB integration).

Builds the production OMOP shape: omop.person (FK target) + omop.condition_occurrence
(surrogate PK, person FK from the identity registry, NO_MAP→0, NULL date). Asserts
FK closure at rest, per-study scoped swap, idempotency, missing-key exclusion, and
that no invalid cross-table state survives a mid-sequence failure + retry.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from sema.compile.compiler_utils import StagingDecision
from sema.compile.fk_closed_compiler import FkClosedCompiler
from sema.compile.fk_closed_compiler_utils import (
    ChildSourceSpec,
    ChildTableSpec,
    ParentTableSpec,
    RegistryJoinSpec,
)
from sema.compile.row_surrogate import surrogate_row_id
from sema.resolve.identity_registry import (
    DEFAULT_SCHEMA,
    DEFAULT_TABLE,
    IdentityRegistry,
)

pytestmark = pytest.mark.unit

_REGISTRY = RegistryJoinSpec(
    schema=DEFAULT_SCHEMA,
    table=DEFAULT_TABLE,
    namespace_column="source_namespace",
    key_column="source_entity_key",
    id_column="entity_id",
)
_PERSON = ParentTableSpec(schema="omop", table="person", id_column="person_id")
_CONDITION = ChildTableSpec(
    schema="omop",
    table="condition_occurrence",
    pk_column="condition_occurrence_id",
    fk_column="person_id",
    value_column="condition_concept_id",
    null_columns=("condition_start_date",),
    row_ref_column="source_row_ref",
    patient_key_column="source_patient_key",
    scope_schema_column="source_schema",
    scope_table_column="source_table",
)

# oncotree code -> concept; ZZZ is a NO_MAP (target NULL -> COALESCE 0, D8).
_DECISIONS = [
    StagingDecision("LUAD", 777926, "RESOLVED", None, "auto_accepted"),
    StagingDecision("GBM", 999, "RESOLVED", None, "auto_accepted"),
    StagingDecision("ZZZ", None, "NO_MAP", "no standard candidate", "auto_accepted"),
]


def _source_spec(schema: str) -> ChildSourceSpec:
    return ChildSourceSpec(
        schema=schema,
        table="sample",
        value_column="oncotree_code",
        row_ref_column="sample_id",
        patient_key_column="patient_id",
    )


def _seed_source(
    conn: duckdb.DuckDBPyConnection, schema: str, rows: list[tuple[str, str | None, str]]
) -> None:
    conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    conn.execute(
        f'CREATE OR REPLACE TABLE "{schema}"."sample" '
        "(sample_id VARCHAR, patient_id VARCHAR, oncotree_code VARCHAR)"
    )
    for sample_id, patient_id, code in rows:
        conn.execute(
            f'INSERT INTO "{schema}"."sample" VALUES (?, ?, ?)',
            [sample_id, patient_id, code],
        )


def _seed_registry(
    conn: duckdb.DuckDBPyConnection, namespace: str, keys: list[str]
) -> None:
    IdentityRegistry(conn).get_or_create(
        [(namespace, k) for k in keys], run_id="s1-06"
    )


@pytest.fixture()
def conn(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(tmp_path / "poc.duckdb"))


def _materialize(conn: duckdb.DuckDBPyConnection, schema: str) -> object:
    return FkClosedCompiler(no_map_default=0).materialize(
        conn,
        parent=_PERSON,
        child=_CONDITION,
        source=_source_spec(schema),
        registry=_REGISTRY,
        decisions=_DECISIONS,
    )


class TestHappyPath:
    def test_fk_closed_person_and_condition(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        _seed_source(
            conn,
            "cbio_msk",
            [("S-1", "P-1", "LUAD"), ("S-2", "P-2", "GBM"), ("S-3", "P-1", "LUAD")],
        )
        _seed_registry(conn, "cbio_msk", ["P-1", "P-2"])
        result = _materialize(conn, "cbio_msk")
        assert result.child_rows == 3
        assert result.parent_rows == 2
        assert result.missing_key_rows == 0
        # every condition person_id references a real person (FK-closed).
        orphans = conn.execute(
            'SELECT COUNT(*) FROM omop.condition_occurrence c '
            'LEFT JOIN omop.person p ON c.person_id = p.person_id '
            "WHERE p.person_id IS NULL"
        ).fetchone()[0]
        assert orphans == 0

    def test_surrogate_pk_matches_reference_and_null_date(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        _seed_source(conn, "cbio_msk", [("S-1", "P-1", "LUAD")])
        _seed_registry(conn, "cbio_msk", ["P-1"])
        _materialize(conn, "cbio_msk")
        row = conn.execute(
            "SELECT condition_occurrence_id, condition_start_date, source_row_ref, "
            "source_patient_key FROM omop.condition_occurrence"
        ).fetchone()
        assert row[0] == surrogate_row_id("cbio_msk", "sample", "S-1")
        assert row[1] is None  # D4: no fabricated date
        assert row[2] == "S-1"
        assert row[3] == "P-1"

    def test_no_map_concept_defaults_to_zero(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        _seed_source(conn, "cbio_msk", [("S-9", "P-1", "ZZZ")])
        _seed_registry(conn, "cbio_msk", ["P-1"])
        _materialize(conn, "cbio_msk")
        concept = conn.execute(
            "SELECT condition_concept_id FROM omop.condition_occurrence"
        ).fetchone()[0]
        assert concept == 0  # D8: NO_MAP -> OMOP concept 0, not NULL/drop

    def test_person_ids_are_registry_entity_ids(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        _seed_source(conn, "cbio_msk", [("S-1", "P-1", "LUAD"), ("S-2", "P-2", "GBM")])
        _seed_registry(conn, "cbio_msk", ["P-1", "P-2"])
        _materialize(conn, "cbio_msk")
        entity_ids = {
            a.entity_id for a in IdentityRegistry(conn).read_all()
        }
        person_ids = {
            r[0] for r in conn.execute("SELECT person_id FROM omop.person").fetchall()
        }
        assert person_ids == entity_ids


class TestMissingKey:
    def test_blank_key_row_excluded_and_counted(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        _seed_source(
            conn,
            "cbio_msk",
            [("S-1", "P-1", "LUAD"), ("S-2", "  ", "GBM"), ("S-3", None, "LUAD")],
        )
        _seed_registry(conn, "cbio_msk", ["P-1"])
        result = _materialize(conn, "cbio_msk")
        # D5: blank-key rows are NOT written FK-invalid; they route to review.
        assert result.child_rows == 1
        assert result.missing_key_rows == 2
        refs = {
            r[0]
            for r in conn.execute(
                "SELECT source_row_ref FROM omop.condition_occurrence"
            ).fetchall()
        }
        assert refs == {"S-1"}


class TestIdempotencyAndScope:
    def test_rerun_is_idempotent(self, conn: duckdb.DuckDBPyConnection) -> None:
        _seed_source(conn, "cbio_msk", [("S-1", "P-1", "LUAD"), ("S-2", "P-2", "GBM")])
        _seed_registry(conn, "cbio_msk", ["P-1", "P-2"])
        first = _materialize(conn, "cbio_msk")
        second = _materialize(conn, "cbio_msk")
        assert first.child_rows == second.child_rows == 2
        total = conn.execute(
            "SELECT COUNT(*) FROM omop.condition_occurrence"
        ).fetchone()[0]
        assert total == 2  # scoped swap replaced, did not duplicate

    def test_second_study_does_not_clobber_first(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        _seed_source(conn, "cbio_a", [("A-1", "P-1", "LUAD")])
        _seed_registry(conn, "cbio_a", ["P-1"])
        _materialize(conn, "cbio_a")
        _seed_source(conn, "cbio_b", [("B-1", "P-1", "GBM")])
        _seed_registry(conn, "cbio_b", ["P-1"])  # same key text, different namespace
        _materialize(conn, "cbio_b")
        scopes = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT source_schema FROM omop.condition_occurrence"
            ).fetchall()
        }
        assert scopes == {"cbio_a", "cbio_b"}
        # cross-namespace same key text -> two distinct persons (no over-collapse).
        assert conn.execute("SELECT COUNT(*) FROM omop.person").fetchone()[0] == 2


class TestMidSequenceFailure:
    def test_parent_first_ordering_survives_child_failure(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        # Study A fully materialized.
        _seed_source(conn, "cbio_a", [("A-1", "P-1", "LUAD")])
        _seed_registry(conn, "cbio_a", ["P-1"])
        _materialize(conn, "cbio_a")
        # Study B: register its persons, then attempt materialize with NO source
        # table -> the child build fails AFTER the parent is rebuilt.
        _seed_registry(conn, "cbio_b", ["P-2"])
        with pytest.raises(duckdb.Error):
            _materialize(conn, "cbio_b")
        # Parent-first ordering: person now holds A+B, A's conditions stay FK-valid,
        # B wrote nothing -> no dangling FK anywhere.
        assert conn.execute("SELECT COUNT(*) FROM omop.person").fetchone()[0] == 2
        orphans = conn.execute(
            "SELECT COUNT(*) FROM omop.condition_occurrence c "
            "LEFT JOIN omop.person p ON c.person_id = p.person_id "
            "WHERE p.person_id IS NULL"
        ).fetchone()[0]
        assert orphans == 0
        b_rows = conn.execute(
            "SELECT COUNT(*) FROM omop.condition_occurrence "
            "WHERE source_schema = 'cbio_b'"
        ).fetchone()[0]
        assert b_rows == 0
        # Retry once the source exists -> B lands, still FK-closed.
        _seed_source(conn, "cbio_b", [("B-1", "P-2", "GBM")])
        result = _materialize(conn, "cbio_b")
        assert result.child_rows == 1
