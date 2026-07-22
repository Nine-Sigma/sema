"""S1-10 — Stage B collapse + rebuild materialization semantics (hermetic DuckDB).

Proves the H1 correction end-to-end: a registry remap does NOT edit persisted
rows, so dedup materializes only by REBUILDING the child through the collapsed
registry. The precise guarantee: ``condition_occurrence_id`` (the row PK) is
stable across dedup, while the row's ``person_id`` FK is recomputed on rebuild;
the duplicate person is retired and the shape stays FK-closed.

Two MSK studies share the same institutional patient ``P-1``; GBM's ``P-1`` is a
different namespace and must never be dragged into the collapse.
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
from sema.resolve.identity_collapse import collapse_identities
from sema.resolve.identity_registry import (
    DEFAULT_SCHEMA,
    DEFAULT_TABLE,
    IdentityRegistry,
)

pytestmark = pytest.mark.unit

_CHORD = "study_msk_chord"
_IMPACT = "study_msk_impact"
_GBM = "study_gbm"
_GROUPING = {_CHORD: "inst_msk", _IMPACT: "inst_msk"}

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
_DECISIONS = [
    StagingDecision("LUAD", 777926, "RESOLVED", None, "auto_accepted"),
    StagingDecision("GBM", 999, "RESOLVED", None, "auto_accepted"),
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
    conn: duckdb.DuckDBPyConnection, schema: str, rows: list[tuple[str, str, str]]
) -> None:
    conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    conn.execute(
        f'CREATE OR REPLACE TABLE "{schema}"."sample" '
        "(sample_id VARCHAR, patient_id VARCHAR, oncotree_code VARCHAR)"
    )
    for row in rows:
        conn.execute(f'INSERT INTO "{schema}"."sample" VALUES (?, ?, ?)', list(row))


def _register_keys(conn: duckdb.DuckDBPyConnection, schema: str) -> None:
    keys = [
        r[0]
        for r in conn.execute(
            f'SELECT DISTINCT patient_id FROM "{schema}"."sample"'
        ).fetchall()
    ]
    IdentityRegistry(conn).get_or_create(
        [(schema, k) for k in keys], run_id="s1-10"
    )


def _materialize(conn: duckdb.DuckDBPyConnection, schema: str) -> object:
    _register_keys(conn, schema)
    return FkClosedCompiler(no_map_default=0).materialize(
        conn,
        parent=_PERSON,
        child=_CONDITION,
        source=_source_spec(schema),
        registry=_REGISTRY,
        decisions=_DECISIONS,
    )


def _rebuild(conn: duckdb.DuckDBPyConnection, schemas: list[str]) -> object:
    return FkClosedCompiler(no_map_default=0).rebuild_after_collapse(
        conn,
        parent=_PERSON,
        child=_CONDITION,
        sources=[_source_spec(s) for s in schemas],
        registry=_REGISTRY,
        decisions=_DECISIONS,
    )


def _person_count(conn: duckdb.DuckDBPyConnection) -> int:
    return conn.execute("SELECT COUNT(*) FROM omop.person").fetchone()[0]


def _orphans(conn: duckdb.DuckDBPyConnection) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM omop.condition_occurrence c "
        "LEFT JOIN omop.person p ON c.person_id = p.person_id "
        "WHERE p.person_id IS NULL"
    ).fetchone()[0]


def _condition_row(
    conn: duckdb.DuckDBPyConnection, schema: str, row_ref: str
) -> tuple[int, int]:
    return conn.execute(
        "SELECT condition_occurrence_id, person_id FROM omop.condition_occurrence "
        "WHERE source_schema = ? AND source_row_ref = ?",
        [schema, row_ref],
    ).fetchone()


@pytest.fixture()
def conn(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(tmp_path / "poc.duckdb"))


@pytest.fixture()
def two_studies(conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    _seed_source(conn, _CHORD, [("C-1", "P-1", "LUAD"), ("C-2", "P-9", "GBM")])
    _seed_source(conn, _IMPACT, [("I-1", "P-1", "LUAD"), ("I-2", "P-8", "GBM")])
    _materialize(conn, _CHORD)
    _materialize(conn, _IMPACT)
    return conn


class TestStageAProducesDuplicatePersons:
    def test_shared_patient_is_two_persons_before_collapse(
        self, two_studies: duckdb.DuckDBPyConnection
    ) -> None:
        # P-1 sequenced in both studies -> 4 persons for 3 real patients.
        assert _person_count(two_studies) == 4
        chord_p1 = _condition_row(two_studies, _CHORD, "C-1")[1]
        impact_p1 = _condition_row(two_studies, _IMPACT, "I-1")[1]
        assert chord_p1 != impact_p1  # the duplication Stage B must remove


class TestCollapseAndRebuild:
    def test_pk_stable_fk_recomputed_orphan_retired(
        self, two_studies: duckdb.DuckDBPyConnection
    ) -> None:
        conn = two_studies
        before_pk, before_fk = _condition_row(conn, _IMPACT, "I-1")
        chord_p1_person = _condition_row(conn, _CHORD, "C-1")[1]

        result = collapse_identities(
            IdentityRegistry(conn), namespace_grouping=_GROUPING
        )
        _rebuild(conn, [_CHORD, _IMPACT])

        assert result.collapsed_person_count == 1
        after_pk, after_fk = _condition_row(conn, _IMPACT, "I-1")
        assert after_pk == before_pk  # row PK stable across dedup
        assert after_fk != before_fk  # person FK recomputed
        assert after_fk == chord_p1_person  # onto the surviving canonical person
        assert _person_count(conn) == 3  # duplicate person retired
        assert _orphans(conn) == 0  # still FK-closed

    def test_condition_row_count_unchanged(
        self, two_studies: duckdb.DuckDBPyConnection
    ) -> None:
        conn = two_studies
        before = conn.execute(
            "SELECT COUNT(*) FROM omop.condition_occurrence"
        ).fetchone()[0]
        collapse_identities(IdentityRegistry(conn), namespace_grouping=_GROUPING)
        _rebuild(conn, [_CHORD, _IMPACT])
        after = conn.execute(
            "SELECT COUNT(*) FROM omop.condition_occurrence"
        ).fetchone()[0]
        assert after == before == 4  # dedup merges persons, never drops events

    def test_gbm_same_key_text_never_collapses(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        _seed_source(conn, _CHORD, [("C-1", "P-1", "LUAD")])
        _seed_source(conn, _GBM, [("G-1", "P-1", "GBM")])  # different namespace
        _materialize(conn, _CHORD)
        _materialize(conn, _GBM)
        collapse_identities(IdentityRegistry(conn), namespace_grouping=_GROUPING)
        _rebuild(conn, [_CHORD, _GBM])
        # GBM's P-1 is a different person; the collapse must leave both standing.
        assert _person_count(conn) == 2
        assert _orphans(conn) == 0
