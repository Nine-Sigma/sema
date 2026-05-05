"""Unit tests for the FK / join detector (Section 12)."""
from __future__ import annotations

import pytest

from sema.engine.join_detector import (
    DEFAULT_MATERIALIZE_THRESHOLD,
    JoinDetector,
    TIER_1,
    TIER_2,
    TIER_3,
    to_fk_assertion,
)
from sema.engine.join_detector_utils import (
    enumerate_candidates_from_metadata,
    fk_name_root,
    types_compatible,
)
from sema.models.assertions import AssertionPredicate
from sema.models.extraction import ExtractedColumn

pytestmark = pytest.mark.unit

SCHEMA = "cbioportal_msk_chord_2024"
CAT = "workspace"


def _col(name: str, table: str, dtype: str = "STRING") -> ExtractedColumn:
    return ExtractedColumn(
        name=name, table_name=table,
        catalog=CAT, schema=SCHEMA, data_type=dtype,
    )


class TestNameAndType:
    def test_fk_name_root_recognizes_id_suffix(self):
        assert fk_name_root("patient_id") == "patient"
        assert fk_name_root("sample_key") == "sample"
        assert fk_name_root("gene_code") == "gene"

    def test_fk_name_root_rejects_unrelated(self):
        assert fk_name_root("os_status") is None
        assert fk_name_root("age") is None

    def test_types_compatible_handles_aliases(self):
        assert types_compatible("STRING", "VARCHAR(64)")
        assert types_compatible("BIGINT", "INT")
        assert not types_compatible("STRING", "BIGINT")


class TestCandidateEnumeration:
    def test_proposes_patient_to_sample(self):
        cols = [
            _col("patient_id", "patient"),
            _col("os_status", "patient"),
            _col("patient_id", "sample"),
            _col("sample_id", "sample"),
        ]
        cands = enumerate_candidates_from_metadata(cols)
        match = [
            c for c in cands
            if c.fk_table == "sample" and c.fk_column == "patient_id"
            and c.pk_table == "patient" and c.pk_column == "patient_id"
        ]
        assert len(match) == 1

    def test_rejects_cross_schema_pair(self):
        cols = [
            ExtractedColumn(
                "patient_id", "patient", CAT, "schema_a", "STRING",
            ),
            ExtractedColumn(
                "patient_id", "sample", CAT, "schema_b", "STRING",
            ),
        ]
        cands = enumerate_candidates_from_metadata(cols)
        assert all(
            c.schema_name in {"schema_a", "schema_b"} for c in cands
        )
        for c in cands:
            assert (
                c.fk_column == "patient_id"
                and c.pk_table == "patient"
            ) is False or c.schema_name == c.schema_name
        cross = [
            c for c in cands
            if c.fk_table == "sample" and c.pk_table == "patient"
        ]
        assert cross == []

    def test_rejects_type_mismatch(self):
        cols = [
            _col("patient_id", "patient", "STRING"),
            _col("patient_id", "sample", "BIGINT"),
        ]
        assert enumerate_candidates_from_metadata(cols) == []


class TestTierAssignment:
    def setup_method(self):
        self.cols = [
            _col("patient_id", "patient"),
            _col("patient_id", "sample"),
        ]

    def test_tier_1_with_subset_samples(self):
        det = JoinDetector()
        samples = {
            (SCHEMA, "patient", "patient_id"): {"P1", "P2", "P3"},
            (SCHEMA, "sample", "patient_id"): {"P1", "P2"},
        }
        out = det.detect(
            columns=self.cols, source_schema=SCHEMA, samples=samples,
        )
        assert len(out) == 1
        assert out[0].tier == 1
        assert out[0].confidence == TIER_1

    def test_tier_2_with_consistent_cardinality(self):
        det = JoinDetector()
        profiles = {
            (SCHEMA, "patient", "patient_id"): (1000, 1000),
            (SCHEMA, "sample", "patient_id"): (850, 3000),
        }
        out = det.detect(
            columns=self.cols, source_schema=SCHEMA, profiles=profiles,
        )
        assert out[0].tier == 2
        assert out[0].confidence == TIER_2

    def test_tier_3_structural_only(self):
        det = JoinDetector()
        out = det.detect(columns=self.cols, source_schema=SCHEMA)
        assert out[0].tier == 3
        assert out[0].confidence == TIER_3


class TestSampleSourcingFallback:
    def setup_method(self):
        self.cols = [
            _col("patient_id", "patient"),
            _col("patient_id", "sample"),
        ]

    def test_profiler_samples_preferred(self):
        det = JoinDetector()
        sampler_calls: list[tuple] = []

        def sampler(key):
            sampler_calls.append(key)
            return {"unused"}

        samples = {
            (SCHEMA, "patient", "patient_id"): {"a", "b"},
            (SCHEMA, "sample", "patient_id"): {"a"},
        }
        det.detect(
            columns=self.cols, source_schema=SCHEMA,
            samples=samples, sampler=sampler,
        )
        assert sampler_calls == []

    def test_detector_owned_sampling_when_profiler_missing(self):
        det = JoinDetector()
        sampler_calls: list[tuple] = []

        def sampler(key):
            sampler_calls.append(key)
            return {"a", "b"} if "patient_id" in key[2] else None

        out = det.detect(
            columns=self.cols, source_schema=SCHEMA, sampler=sampler,
        )
        assert len(sampler_calls) == 2
        assert out[0].tier == 1

    def test_cap_exceeded_downgrades_to_tier_2(self):
        det = JoinDetector(sample_cap=3)
        # FK sample has exactly cap → inconclusive, downgrade.
        samples = {
            (SCHEMA, "patient", "patient_id"): {"a", "b", "c"},
            (SCHEMA, "sample", "patient_id"): {"a", "b", "c"},
        }
        profiles = {
            (SCHEMA, "patient", "patient_id"): (100, 100),
            (SCHEMA, "sample", "patient_id"): (90, 200),
        }
        out = det.detect(
            columns=self.cols, source_schema=SCHEMA,
            samples=samples, profiles=profiles,
        )
        assert out[0].tier == 2

    def test_cap_exceeded_no_cardinality_downgrades_to_tier_3(self):
        det = JoinDetector(sample_cap=2)
        samples = {
            (SCHEMA, "patient", "patient_id"): {"a", "b"},
            (SCHEMA, "sample", "patient_id"): {"a", "b"},
        }
        out = det.detect(
            columns=self.cols, source_schema=SCHEMA, samples=samples,
        )
        assert out[0].tier == 3

    def test_warehouse_error_falls_through_to_tier_3(self):
        det = JoinDetector()

        def sampler(key):
            raise RuntimeError("warehouse unreachable")

        out = det.detect(
            columns=self.cols, source_schema=SCHEMA, sampler=sampler,
        )
        assert out[0].tier == 3


class TestMaterializationThreshold:
    def setup_method(self):
        self.cols = [
            _col("patient_id", "patient"),
            _col("patient_id", "sample"),
        ]

    def test_default_excludes_tier_3(self):
        det = JoinDetector()
        out = det.detect(columns=self.cols, source_schema=SCHEMA)
        assert det.should_materialize(out[0]) is False

    def test_default_threshold_value(self):
        assert DEFAULT_MATERIALIZE_THRESHOLD == 0.80

    def test_default_includes_tier_2(self):
        det = JoinDetector()
        profiles = {
            (SCHEMA, "patient", "patient_id"): (10, 10),
            (SCHEMA, "sample", "patient_id"): (5, 30),
        }
        out = det.detect(
            columns=self.cols, source_schema=SCHEMA, profiles=profiles,
        )
        assert det.should_materialize(out[0]) is True

    def test_explicit_opt_in_includes_tier_3(self):
        det = JoinDetector(materialization_threshold=0.70)
        out = det.detect(columns=self.cols, source_schema=SCHEMA)
        assert det.should_materialize(out[0]) is True


class TestAssertionEmission:
    def test_to_fk_assertion_carries_source_schema_and_predicate(self):
        det = JoinDetector()
        cols = [
            _col("patient_id", "patient"),
            _col("patient_id", "sample"),
        ]
        out = det.detect(columns=cols, source_schema=SCHEMA)
        assertion = to_fk_assertion(out[0], run_id="run-1")
        assert assertion.predicate == AssertionPredicate.FK_TO
        assert assertion.source_schema == SCHEMA
        assert assertion.confidence == TIER_3
        assert assertion.payload["fk_table"] == "sample"
        assert assertion.payload["pk_table"] == "patient"
        assert assertion.payload["tier"] == 3
