"""Tests for WarehouseProfiler heuristic domain detection."""

import pytest

from sema.models.extraction import ExtractedColumn, ExtractedTable
from sema.pipeline.profiler import WarehouseProfiler

pytestmark = pytest.mark.unit


@pytest.fixture
def profiler() -> WarehouseProfiler:
    return WarehouseProfiler()


def _tables(*names: str) -> list[ExtractedTable]:
    return [
        ExtractedTable(name=n, catalog="cat", schema="sch")
        for n in names
    ]


def _columns(
    table: str, *col_names: str,
) -> list[ExtractedColumn]:
    return [
        ExtractedColumn(
            name=c, table_name=table,
            catalog="cat", schema="sch", data_type="STRING",
        )
        for c in col_names
    ]


class TestHealthcareDominant:
    def test_healthcare_tables(self, profiler: WarehouseProfiler) -> None:
        tables = _tables(
            "patient_demographics", "encounter_detail",
            "diagnosis_history", "provider_registry",
            "medication_orders", "lab_results",
        )
        columns = _columns("patient_demographics", "patient_id", "admission_date")
        profile = profiler.profile(tables, columns, "ds1", "run-1")
        assert profile.primary_domain == "healthcare"
        assert profile.domains.get("healthcare", 0) > 0.5
        assert profile.confidence >= 0.6

    def test_high_confidence_when_dominant(
        self, profiler: WarehouseProfiler,
    ) -> None:
        tables = _tables(
            "patient", "encounter", "diagnosis",
            "procedure", "claim", "provider",
            "medication", "prescription", "vitals", "lab",
        )
        profile = profiler.profile(tables, [], "ds1", "run-1")
        assert profile.confidence >= 0.8


class TestMixedDomain:
    def test_healthcare_plus_financial(
        self, profiler: WarehouseProfiler,
    ) -> None:
        tables = _tables(
            "patient", "encounter", "diagnosis",
            "account", "transaction", "payment",
        )
        profile = profiler.profile(tables, [], "ds1", "run-1")
        assert "healthcare" in profile.domains
        assert "financial" in profile.domains
        # Neither should be overwhelming
        assert profile.domains["healthcare"] < 0.9
        assert profile.domains["financial"] < 0.9


class TestUnknownDomain:
    def test_no_signals(self, profiler: WarehouseProfiler) -> None:
        tables = _tables("foo_bar", "baz_qux", "alpha_beta")
        columns = _columns("foo_bar", "x", "y", "z")
        profile = profiler.profile(tables, columns, "ds1", "run-1")
        assert profile.confidence <= 0.5

    def test_empty_inputs(self, profiler: WarehouseProfiler) -> None:
        profile = profiler.profile([], [], "ds1", "run-1")
        assert profile.confidence <= 0.3
        assert profile.primary_domain is None


class TestRealEstateDominant:
    def test_proptech_warehouse(
        self, profiler: WarehouseProfiler,
    ) -> None:
        tables = _tables(
            "property_listing", "building_details",
            "lease_agreement", "tenant_info",
            "parcel_data", "address_lookup",
        )
        columns = _columns(
            "property_listing",
            "listing_id", "sqft", "bedroom", "bathroom",
            "zoning", "appraisal",
        )
        profile = profiler.profile(tables, columns, "ds1", "run-1")
        assert profile.primary_domain == "real_estate"
        assert profile.domains.get("real_estate", 0) > 0.5
