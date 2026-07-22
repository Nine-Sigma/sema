"""S1-10 — Stage B pipeline entry (collapse → rebuild → Gate-D-lite) end-to-end.

Runs Stage A for two MSK studies sharing patient ``P-1`` through the real fit
chain, then the single Stage B call, and asserts the collapsed row-level state:
one fewer person, per-study Gate-D-lite still green, idempotent replay.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import duckdb
import pytest

from sema.compile.compiler_utils import StagingDecision
from sema.compile.fk_backend import DUCKDB_FK_BACKEND
from sema.compile.fk_closed_compiler_utils import ChildSourceSpec, RegistryJoinSpec
from sema.pipeline.fk_closed_fit import (
    FkClosedFitRequest,
    run_fk_closed_fit,
    run_stage_b_collapse,
)
from sema.resolve.identity_registry import (
    DEFAULT_SCHEMA,
    DEFAULT_TABLE,
    IdentityRegistry,
)
from showcase.cbioportal_to_omop.omop_policy import MISSING_PERSON_KEY_REASON, make_omop_fk_specs

pytestmark = pytest.mark.unit

_PARENT, _CHILD, _REQUIRED = make_omop_fk_specs("omop_stage_a")
_REGISTRY_SPEC = RegistryJoinSpec(
    schema=DEFAULT_SCHEMA,
    table=DEFAULT_TABLE,
    namespace_column="source_namespace",
    key_column="source_entity_key",
    id_column="entity_id",
)
_DECISIONS = [
    StagingDecision("LUAD", 777926, "RESOLVED", None, "auto_accepted"),
    StagingDecision("GBM", 4000, "RESOLVED", None, "auto_accepted"),
]
_CHORD = "cbioportal_msk_chord_2024"
_IMPACT = "cbioportal_msk_impact_2026"
_GROUPING = {_CHORD: "msk_dmp", _IMPACT: "msk_dmp"}


def _source(schema: str) -> ChildSourceSpec:
    return ChildSourceSpec(
        schema=schema,
        table="sample",
        value_column="oncotree_code",
        row_ref_column="sample_id",
        patient_key_column="patient_id",
    )


def _request(schema: str, keys: list[str], row_count: int) -> FkClosedFitRequest:
    return FkClosedFitRequest(
        source=_source(schema),
        source_row_count=row_count,
        distinct_patient_keys=keys,
        parent=_PARENT,
        child=_CHILD,
        registry_spec=_REGISTRY_SPEC,
        decisions=_DECISIONS,
        required_fields=_REQUIRED,
        no_map_default=0,
        missing_key_reason=MISSING_PERSON_KEY_REASON,
        run_id="s1-10",
    )


def _seed(conn: duckdb.DuckDBPyConnection, schema: str, rows: list[tuple[str, str, str]]) -> None:
    conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    conn.execute(
        f'CREATE OR REPLACE TABLE "{schema}"."sample" '
        "(sample_id VARCHAR, patient_id VARCHAR, oncotree_code VARCHAR)"
    )
    conn.executemany(f'INSERT INTO "{schema}"."sample" VALUES (?, ?, ?)', rows)


def _person_count(conn: duckdb.DuckDBPyConnection) -> int:
    return conn.execute("SELECT COUNT(*) FROM omop_stage_a.person").fetchone()[0]


@pytest.fixture()
def staged(tmp_path: Path) -> tuple[duckdb.DuckDBPyConnection, list[FkClosedFitRequest]]:
    conn = duckdb.connect(str(tmp_path / "poc.duckdb"))
    _seed(conn, _CHORD, [("C-1", "P-1", "LUAD"), ("C-2", "P-9", "GBM")])
    _seed(conn, _IMPACT, [("I-1", "P-1", "LUAD"), ("I-2", "P-8", "GBM")])
    requests = [
        _request(_CHORD, ["P-1", "P-9"], 2),
        _request(_IMPACT, ["P-1", "P-8"], 2),
    ]
    registry = IdentityRegistry(conn)
    for req in requests:
        run_fk_closed_fit(req, registry=registry, target_cursor=conn)
    return conn, requests


class TestStageBPipeline:
    def test_collapse_retires_duplicate_person_and_passes_gate(
        self, staged: tuple[duckdb.DuckDBPyConnection, list[FkClosedFitRequest]]
    ) -> None:
        conn, requests = staged
        assert _person_count(conn) == 4  # Stage A duplicated the shared MSK patient

        result = run_stage_b_collapse(
            requests,
            namespace_grouping=_GROUPING,
            registry=IdentityRegistry(conn),
            target_cursor=conn,
            backend=DUCKDB_FK_BACKEND,
        )

        assert result.collapse.collapsed_person_count == 1
        assert result.fk.parent_rows == 3  # 4 → 3 after dedup
        assert result.fk.child_rows == 4  # events preserved
        assert all(qa.passed for qa in result.qa)  # per-study Gate-D-lite green
        assert _person_count(conn) == 3

    def test_stage_b_is_idempotent(
        self, staged: tuple[duckdb.DuckDBPyConnection, list[FkClosedFitRequest]]
    ) -> None:
        conn, requests = staged
        kwargs = dict(
            namespace_grouping=_GROUPING,
            target_cursor=conn,
            backend=DUCKDB_FK_BACKEND,
        )
        run_stage_b_collapse(requests, registry=IdentityRegistry(conn), **kwargs)
        second = run_stage_b_collapse(
            requests, registry=IdentityRegistry(conn), **kwargs
        )
        assert second.collapse.collapsed_person_count == 0
        assert _person_count(conn) == 3

    def test_no_grouping_leaves_all_persons(
        self, staged: tuple[duckdb.DuckDBPyConnection, list[FkClosedFitRequest]]
    ) -> None:
        conn, requests = staged
        result = run_stage_b_collapse(
            requests,
            namespace_grouping={},
            registry=IdentityRegistry(conn),
            target_cursor=conn,
        )
        assert result.collapse.collapsed_person_count == 0
        assert _person_count(conn) == 4

    def test_divergent_shape_rejected(
        self, staged: tuple[duckdb.DuckDBPyConnection, list[FkClosedFitRequest]]
    ) -> None:
        conn, requests = staged
        # rebuild borrows the head's decisions/no_map_default for ALL sources, so
        # a request that disagrees would be silently rewritten under the head's
        # policy. The guard must reject it rather than the first request winning.
        divergent = dataclasses.replace(requests[1], no_map_default=999)
        with pytest.raises(ValueError, match="same shape"):
            run_stage_b_collapse(
                [requests[0], divergent],
                namespace_grouping=_GROUPING,
                registry=IdentityRegistry(conn),
                target_cursor=conn,
            )
