"""US-005 — schema drift guard + real-code round-trip (integration).

Freezes the §1.5(a) contract: the live DuckDB table's column set must equal
the frozen list EXACTLY — adding/removing/renaming a column fails here, which
is the mechanism that keeps every downstream story aligned. Runs against a
temp DuckDB file (no external warehouse), so it is always executable under
``-m integration``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.models.planner.lifecycle import Status
from sema.resolve.value_mapping_store import open_duckdb_value_mapping_store
from sema.resolve.value_mapping_store_utils import (
    FROZEN_COLUMNS,
    ResolutionStatus,
    ValueMapping,
)

pytestmark = pytest.mark.integration


def _row(code: str, concept_id: int) -> ValueMapping:
    return ValueMapping(
        source_vocabulary="OncoTree",
        normalized_source_value=code,
        target_property_ref="omop.condition_occurrence.condition_concept_id",
        target_field="condition_concept_id",
        vocab_binding="SNOMED",
        concept_id=concept_id,
        vocab_release="2024-Q1",
        valid_start="2002-01-31",
        valid_end="2099-12-31",
        resolution_status=ResolutionStatus.RESOLVED,
        no_map_reason=None,
        confidence=0.99,
        status=Status.auto_accepted,
        resolver_policy_ref="omop.oncotree_to_snomed_condition",
        run_id="run-int",
    )


def test_live_table_columns_equal_frozen_list_exactly(tmp_path: Path) -> None:
    store = open_duckdb_value_mapping_store(str(tmp_path / "drift.duckdb"))
    try:
        assert tuple(store.column_names()) == FROZEN_COLUMNS
    finally:
        store.close()


def test_round_trip_real_resolved_codes(tmp_path: Path) -> None:
    store = open_duckdb_value_mapping_store(str(tmp_path / "rt.duckdb"))
    try:
        # Representative OncoTree→SNOMED concept_ids (US-003 verified LUAD→777926).
        store.upsert([_row("LUAD", 777926), _row("GBM", 372608), _row("BRCA", 4112853)])
        by_code = {r.normalized_source_value: r for r in store.read_all()}
        assert by_code["LUAD"].concept_id == 777926
        assert set(by_code) == {"LUAD", "GBM", "BRCA"}
        assert tuple(store.column_names()) == FROZEN_COLUMNS
    finally:
        store.close()
