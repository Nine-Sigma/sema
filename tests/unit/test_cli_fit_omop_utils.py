"""S1-08 CLI helpers: identity-source enum + scoped decision loading."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from sema.cli_fit_omop_utils import (
    enumerate_identity_source_duckdb,
    load_staging_decisions,
)
from sema.models.planner.lifecycle import Status
from sema.resolve.value_mapping_store import ValueMappingStore
from sema.resolve.value_mapping_store_utils import ResolutionStatus, ValueMapping

pytestmark = pytest.mark.unit


def _mapping(code: str, concept: int | None, policy: str) -> ValueMapping:
    no_map = concept is None
    return ValueMapping(
        source_vocabulary="OncoTree",
        normalized_source_value=code,
        target_property_ref="target.condition_occurrence_staging.condition_concept_id",
        target_field="condition_concept_id",
        vocab_binding="OMOP-Condition",
        concept_id=concept,
        vocab_release="omop-vocab-2024",
        valid_start=None,
        valid_end=None,
        resolution_status=ResolutionStatus.NO_MAP if no_map else ResolutionStatus.RESOLVED,
        no_map_reason="no standard candidate" if no_map else None,
        confidence=1.0,
        status=Status.auto_accepted,
        resolver_policy_ref=policy,
        run_id="r1",
    )


def test_enumerate_identity_source_counts_all_rows_but_distinct_nonblank_keys(
    tmp_path: Path,
) -> None:
    conn = duckdb.connect(str(tmp_path / "poc.duckdb"))
    conn.execute('CREATE SCHEMA "s"')
    conn.execute('CREATE TABLE "s"."sample" (sid VARCHAR, pid VARCHAR)')
    conn.executemany(
        'INSERT INTO "s"."sample" VALUES (?, ?)',
        [("S1", "P-1"), ("S2", "P-2"), ("S3", "P-1"), ("S4", "  "), ("S5", None)],
    )
    keys, row_count = enumerate_identity_source_duckdb(
        conn, schema="s", table="sample", patient_key_column="pid"
    )
    assert keys == ["P-1", "P-2"]  # distinct, non-blank
    assert row_count == 5  # ALL rows (identity denominator includes blanks)


def test_load_staging_decisions_scopes_to_policy_ref(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "poc.duckdb"))
    store = ValueMappingStore(conn)
    store.upsert(
        [
            _mapping("LUAD", 777926, "omop.oncotree_condition"),
            _mapping("ZZZ", None, "omop.oncotree_condition"),
            _mapping("OLD", 111, "omop.oncotree_to_snomed_condition"),  # stale ref
        ]
    )
    decisions = load_staging_decisions(conn, policy_ref="omop.oncotree_condition")
    codes = {d.normalized_source_value for d in decisions}
    assert codes == {"LUAD", "ZZZ"}  # the stale-ref row is excluded
