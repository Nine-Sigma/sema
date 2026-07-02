"""US-005 — value-mapping store unit tests (hermetic, temp DuckDB).

Freezes the §1.5(a) frozen column list and the location-independent grain:
one row per distinct source code per target binding; a re-write for the same
grain key upserts, never duplicates; NO_MAP rows persist concept_id=NULL +
no_map_reason; lifecycle ``status`` round-trips.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.models.planner.lifecycle import Status
from sema.resolve.value_mapping_store import (
    ValueMappingStore,
    open_duckdb_value_mapping_store,
)
from sema.resolve.value_mapping_store_utils import (
    FROZEN_COLUMNS,
    GRAIN_KEY,
    ResolutionStatus,
    ValueMapping,
)

pytestmark = pytest.mark.unit


# §1.5(a) frozen column list — restated literally ONLY here as the drift anchor.
_SPEC_COLUMNS = (
    "source_vocabulary",
    "normalized_source_value",
    "target_property_ref",
    "target_field",
    "vocab_binding",
    "concept_id",
    "vocab_release",
    "valid_start",
    "valid_end",
    "resolution_status",
    "no_map_reason",
    "confidence",
    "status",
    "resolver_policy_ref",
    "run_id",
)

_SPEC_GRAIN = (
    "source_vocabulary",
    "normalized_source_value",
    "target_property_ref",
    "resolver_policy_ref",
    "vocab_release",
)


def _resolved(
    code: str = "LUAD",
    concept_id: int = 777926,
    *,
    vocab_release: str = "2024-Q1",
    status: Status = Status.auto_accepted,
    confidence: float = 0.99,
) -> ValueMapping:
    return ValueMapping(
        source_vocabulary="OncoTree",
        normalized_source_value=code,
        target_property_ref="omop.condition_occurrence.condition_concept_id",
        target_field="condition_concept_id",
        vocab_binding="SNOMED",
        concept_id=concept_id,
        vocab_release=vocab_release,
        valid_start="2002-01-31",
        valid_end="2099-12-31",
        resolution_status=ResolutionStatus.RESOLVED,
        no_map_reason=None,
        confidence=confidence,
        status=status,
        resolver_policy_ref="omop.oncotree_to_snomed_condition",
        run_id="run-1",
    )


def _no_map(code: str = "ZZZ", *, reason: str = "no standard candidate") -> ValueMapping:
    return ValueMapping(
        source_vocabulary="OncoTree",
        normalized_source_value=code,
        target_property_ref="omop.condition_occurrence.condition_concept_id",
        target_field="condition_concept_id",
        vocab_binding="SNOMED",
        concept_id=None,
        vocab_release="2024-Q1",
        valid_start=None,
        valid_end=None,
        resolution_status=ResolutionStatus.NO_MAP,
        no_map_reason=reason,
        confidence=0.0,
        status=Status.auto_accepted,
        resolver_policy_ref="omop.oncotree_to_snomed_condition",
        run_id="run-1",
    )


@pytest.fixture()
def store(tmp_path: Path) -> ValueMappingStore:
    return open_duckdb_value_mapping_store(str(tmp_path / "vms.duckdb"))


class TestFrozenSchema:
    def test_frozen_columns_constant_matches_spec(self) -> None:
        assert FROZEN_COLUMNS == _SPEC_COLUMNS

    def test_grain_key_constant_matches_spec(self) -> None:
        assert GRAIN_KEY == _SPEC_GRAIN

    def test_created_table_columns_equal_frozen_list_exactly(
        self, store: ValueMappingStore
    ) -> None:
        assert tuple(store.column_names()) == FROZEN_COLUMNS

    def test_column_names_scoped_to_current_catalog(self, tmp_path: Path) -> None:
        # A second attached catalog that ALSO has sema_resolve.value_mapping must
        # not double the column list: column_names() reflects the store's own
        # (current) catalog, matching its unqualified read/write SQL.
        import duckdb

        other = tmp_path / "other.duckdb"
        oc = duckdb.connect(str(other))
        ValueMappingStore(oc)
        oc.close()
        conn = duckdb.connect(str(tmp_path / "work.duckdb"))
        store = ValueMappingStore(conn)
        conn.execute(f"ATTACH '{other}' AS poc")
        assert tuple(store.column_names()) == FROZEN_COLUMNS

    def test_store_is_location_independent(self) -> None:
        assert "source_schema" not in FROZEN_COLUMNS
        assert "source_table" not in FROZEN_COLUMNS


class TestRoundTrip:
    def test_write_and_read_back_distinct_codes(self, store: ValueMappingStore) -> None:
        decisions = [_resolved("LUAD", 1), _resolved("GBM", 2), _resolved("BRCA", 3)]
        store.upsert(decisions)
        read = {d.normalized_source_value: d for d in store.read_all()}
        assert set(read) == {"LUAD", "GBM", "BRCA"}
        assert read["GBM"] == _resolved("GBM", 2)

    def test_get_by_grain_key(self, store: ValueMappingStore) -> None:
        store.upsert([_resolved("LUAD", 777926)])
        got = store.get(
            source_vocabulary="OncoTree",
            normalized_source_value="LUAD",
            target_property_ref="omop.condition_occurrence.condition_concept_id",
            resolver_policy_ref="omop.oncotree_to_snomed_condition",
            vocab_release="2024-Q1",
        )
        assert got is not None
        assert got.concept_id == 777926

    def test_get_missing_returns_none(self, store: ValueMappingStore) -> None:
        assert (
            store.get(
                source_vocabulary="OncoTree",
                normalized_source_value="NOPE",
                target_property_ref="x",
                resolver_policy_ref="y",
                vocab_release="z",
            )
            is None
        )


class TestGrainEnforcement:
    def test_second_write_same_grain_upserts(self, store: ValueMappingStore) -> None:
        store.upsert([_resolved("LUAD", 1, confidence=0.5)])
        store.upsert([_resolved("LUAD", 9, confidence=0.95)])
        rows = store.read_all()
        assert len(rows) == 1
        assert rows[0].concept_id == 9
        assert rows[0].confidence == 0.95

    def test_status_change_upserts_same_grain(self, store: ValueMappingStore) -> None:
        store.upsert([_resolved("LUAD", 1, status=Status.candidate)])
        store.upsert([_resolved("LUAD", 1, status=Status.human_pinned)])
        rows = store.read_all()
        assert len(rows) == 1
        assert rows[0].status is Status.human_pinned

    def test_different_vocab_release_is_distinct_row(
        self, store: ValueMappingStore
    ) -> None:
        store.upsert([_resolved("LUAD", 1, vocab_release="2024-Q1")])
        store.upsert([_resolved("LUAD", 2, vocab_release="2024-Q2")])
        assert store.count() == 2


class TestNoMapAndStatus:
    def test_no_map_row_persists_null_concept_and_reason(
        self, store: ValueMappingStore
    ) -> None:
        store.upsert([_no_map("ZZZ", reason="dead end")])
        row = store.read_all()[0]
        assert row.resolution_status is ResolutionStatus.NO_MAP
        assert row.concept_id is None
        assert row.no_map_reason == "dead end"

    @pytest.mark.parametrize(
        "status",
        [
            Status.candidate,
            Status.auto_accepted,
            Status.review_pending,
            Status.human_pinned,
            Status.rejected,
        ],
    )
    def test_status_round_trips(self, store: ValueMappingStore, status: Status) -> None:
        store.upsert([_resolved("LUAD", 1, status=status)])
        assert store.read_all()[0].status is status


class TestValueMappingValidation:
    def test_resolved_requires_concept_id(self) -> None:
        with pytest.raises(ValueError, match="concept_id"):
            ValueMapping(
                source_vocabulary="OncoTree",
                normalized_source_value="LUAD",
                target_property_ref="p",
                target_field="condition_concept_id",
                vocab_binding="SNOMED",
                concept_id=None,
                vocab_release="2024-Q1",
                valid_start=None,
                valid_end=None,
                resolution_status=ResolutionStatus.RESOLVED,
                no_map_reason=None,
                confidence=0.9,
                status=Status.auto_accepted,
                resolver_policy_ref="r",
                run_id="run-1",
            )

    def test_no_map_forbids_concept_id(self) -> None:
        with pytest.raises(ValueError, match="NO_MAP"):
            ValueMapping(
                source_vocabulary="OncoTree",
                normalized_source_value="LUAD",
                target_property_ref="p",
                target_field="condition_concept_id",
                vocab_binding="SNOMED",
                concept_id=5,
                vocab_release="2024-Q1",
                valid_start=None,
                valid_end=None,
                resolution_status=ResolutionStatus.NO_MAP,
                no_map_reason="x",
                confidence=0.0,
                status=Status.auto_accepted,
                resolver_policy_ref="r",
                run_id="run-1",
            )

    def test_no_map_requires_reason(self) -> None:
        with pytest.raises(ValueError, match="no_map_reason"):
            ValueMapping(
                source_vocabulary="OncoTree",
                normalized_source_value="LUAD",
                target_property_ref="p",
                target_field="condition_concept_id",
                vocab_binding="SNOMED",
                concept_id=None,
                vocab_release="2024-Q1",
                valid_start=None,
                valid_end=None,
                resolution_status=ResolutionStatus.NO_MAP,
                no_map_reason=None,
                confidence=0.0,
                status=Status.auto_accepted,
                resolver_policy_ref="r",
                run_id="run-1",
            )

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            _resolved("LUAD", 1, confidence=1.5)
