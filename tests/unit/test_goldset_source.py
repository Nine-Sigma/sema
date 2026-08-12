"""US-002 / G-01: the gold set enumerates a DECLARED scope, never a discovered one.

``source_of_truth`` is an executable source specification, not a schema list:
raw cbioportal ``sample`` tables and ``sema_staging.condition_staging`` are
different shapes (per-schema table vs one table keyed by a ``source_schema``
column), so a schema list alone cannot address both.

Backed by an in-memory DuckDB fixture, so the contract holds off the one
developer machine that has ``~/.sema/poc.duckdb``.
"""

from __future__ import annotations

import duckdb
import pytest

from sema.eval.goldset_source import (
    SourceKind,
    SourceSpec,
    discover_oncotree_schemas,
    enumerate_scoped_codes,
    scoped_enumeration_sql,
)

pytestmark = pytest.mark.unit


_RAW_SAMPLES = {
    "cbioportal_study_a": ["LUAD", "LUAD", "COAD", None, "  "],
    "cbioportal_study_b": ["LUAD", "GBM"],
    "cbioportal_study_c": ["IDC", "IDC", "IDC"],
}

_STAGING = [
    ("cbioportal_study_a", "LUAD"),
    ("cbioportal_study_a", "LUAD"),
    ("cbioportal_study_a", "COAD"),
    ("cbioportal_study_b", "LUAD"),
    ("cbioportal_study_b", "GBM"),
    ("cbioportal_study_c", "IDC"),
]


@pytest.fixture()
def con() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    for schema, codes in _RAW_SAMPLES.items():
        conn.execute(f"CREATE SCHEMA {schema}")
        conn.execute(f"CREATE TABLE {schema}.sample (ONCOTREE_CODE VARCHAR)")
        for code in codes:
            conn.execute(f"INSERT INTO {schema}.sample VALUES (?)", [code])
    conn.execute("CREATE SCHEMA sema_staging")
    conn.execute(
        "CREATE TABLE sema_staging.condition_staging "
        "(source_schema VARCHAR, source_oncotree_code VARCHAR)"
    )
    for row in _STAGING:
        conn.execute("INSERT INTO sema_staging.condition_staging VALUES (?, ?)", list(row))
    try:
        yield conn
    finally:
        conn.close()


def _raw_spec(*schemas: str) -> SourceSpec:
    return SourceSpec(
        kind=SourceKind.RAW_SAMPLES,
        table="sample",
        code_column="ONCOTREE_CODE",
        scope_values=tuple(schemas),
    )


def _staging_spec(*schemas: str) -> SourceSpec:
    return SourceSpec(
        kind=SourceKind.STAGING,
        table="sema_staging.condition_staging",
        code_column="source_oncotree_code",
        scope_column="source_schema",
        scope_values=tuple(schemas),
    )


def test_declared_scope_ignores_an_unlisted_study(con) -> None:  # type: ignore[no-untyped-def]
    """The drift that broke the suite: ingesting a study must not widen scope."""
    counts = dict(enumerate_scoped_codes(con, _raw_spec("cbioportal_study_a", "cbioportal_study_b")))

    assert counts == {"LUAD": 3, "COAD": 1, "GBM": 1}
    assert "IDC" not in counts, "an unlisted study leaked into the declared scope"


def test_both_kinds_agree_on_the_same_declared_scope(con) -> None:  # type: ignore[no-untyped-def]
    """G-01's test: same scope, two shapes, one code set (modulo unstaged studies)."""
    scope = ("cbioportal_study_a", "cbioportal_study_b")
    raw = dict(enumerate_scoped_codes(con, _raw_spec(*scope)))
    staging = dict(enumerate_scoped_codes(con, _staging_spec(*scope)))

    assert raw == staging


def test_staging_scope_drops_a_study_absent_from_staging(con) -> None:  # type: ignore[no-untyped-def]
    """The live GBM delta: a study in raw but not staged simply leaves the scope."""
    raw = dict(enumerate_scoped_codes(con, _raw_spec("cbioportal_study_b")))
    con.execute("DELETE FROM sema_staging.condition_staging WHERE source_oncotree_code = 'GBM'")
    staging = dict(enumerate_scoped_codes(con, _staging_spec("cbioportal_study_b")))

    assert set(raw) - set(staging) == {"GBM"}


def test_enumeration_excludes_null_and_blank_codes(con) -> None:  # type: ignore[no-untyped-def]
    counts = dict(enumerate_scoped_codes(con, _raw_spec("cbioportal_study_a")))

    assert counts == {"LUAD": 2, "COAD": 1}


def test_enumeration_is_ordered_by_descending_count_then_code(con) -> None:  # type: ignore[no-untyped-def]
    """Tier construction depends on this order being total and deterministic."""
    spec = _staging_spec("cbioportal_study_a", "cbioportal_study_b", "cbioportal_study_c")

    assert enumerate_scoped_codes(con, spec) == [
        ("LUAD", 3),
        ("COAD", 1),
        ("GBM", 1),
        ("IDC", 1),
    ]


def test_empty_scope_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        scoped_enumeration_sql(_raw_spec())


@pytest.mark.parametrize(
    "spec",
    [
        SourceSpec(
            kind=SourceKind.RAW_SAMPLES,
            table="sample; DROP TABLE x",
            code_column="ONCOTREE_CODE",
            scope_values=("cbioportal_study_a",),
        ),
        SourceSpec(
            kind=SourceKind.RAW_SAMPLES,
            table="sample",
            code_column="ONCOTREE_CODE",
            scope_values=("a'; DROP TABLE x --",),
        ),
        SourceSpec(
            kind=SourceKind.STAGING,
            table="sema_staging.condition_staging",
            code_column="source_oncotree_code",
            scope_column="source_schema--",
            scope_values=("cbioportal_study_a",),
        ),
    ],
)
def test_identifiers_are_validated_at_the_boundary(spec: SourceSpec) -> None:
    with pytest.raises(ValueError, match="identifier"):
        scoped_enumeration_sql(spec)


def test_staging_kind_requires_a_scope_column() -> None:
    spec = SourceSpec(
        kind=SourceKind.STAGING,
        table="sema_staging.condition_staging",
        code_column="source_oncotree_code",
        scope_values=("cbioportal_study_a",),
    )

    with pytest.raises(ValueError, match="scope_column"):
        scoped_enumeration_sql(spec)


def test_spec_round_trips_through_json() -> None:
    spec = _staging_spec("cbioportal_study_a")

    assert SourceSpec.from_dict(spec.as_dict()) == spec


def test_discovery_still_available_for_refresh(con) -> None:  # type: ignore[no-untyped-def]
    """Auto-discovery survives — but only as a refresh input, never at test time."""
    assert discover_oncotree_schemas(con) == [
        "cbioportal_study_a",
        "cbioportal_study_b",
        "cbioportal_study_c",
    ]
