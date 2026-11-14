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
        ref=f"databricks://host/{CAT}/{SCHEMA}/{table}/{name}",
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
                ref="ref-a",
            ),
            ExtractedColumn(
                "patient_id", "sample", CAT, "schema_b", "STRING",
                ref="ref-b",
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

    def test_proposes_same_named_key_whose_table_is_not_named_after_it(self):
        cols = [
            _col("visit_id", "visit_header"),
            _col("visit_id", "visit_line"),
        ]
        pairs = {
            (c.fk_table, c.pk_table)
            for c in enumerate_candidates_from_metadata(cols)
        }
        assert ("visit_line", "visit_header") in pairs

    def test_proposes_role_prefixed_keys(self):
        cols = [
            _col("site_id", "site"),
            _col("origin_site_id", "move"),
            _col("target_site_id", "move"),
        ]
        pairs = {
            (c.fk_column, c.pk_table, c.pk_column)
            for c in enumerate_candidates_from_metadata(cols)
        }
        assert ("origin_site_id", "site", "site_id") in pairs
        assert ("target_site_id", "site", "site_id") in pairs

    def test_bare_key_names_propose_nothing_across_tables(self):
        cols = [
            _col("id", "site"),
            _col("id", "move"),
            _col("site_id", "visit"),
        ]
        pairs = {
            (c.fk_table, c.fk_column, c.pk_table)
            for c in enumerate_candidates_from_metadata(cols)
        }
        assert pairs == {("visit", "site_id", "site")}

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

    def test_tier_1_when_containment_covers_the_referencing_sample(self):
        det = JoinDetector()
        out = det.detect(
            columns=self.cols, source_schema=SCHEMA,
            sampler=lambda key: {"P1", "P2"},
            containment=lambda key, values: len(values),
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

    def test_refuted_containment_emits_nothing_even_when_cardinality_fits(self):
        det = JoinDetector()
        profiles = {
            (SCHEMA, "patient", "patient_id"): (1000, 1000),
            (SCHEMA, "sample", "patient_id"): (10, 30),
        }
        out = det.detect(
            columns=self.cols, source_schema=SCHEMA, profiles=profiles,
            sampler=lambda key: {"X1", "X2", "X3", "X4", "X5"},
            containment=lambda key, values: 1,
        )
        assert out == []

    def test_non_unique_referenced_column_is_never_a_candidate(self):
        det = JoinDetector(materialization_threshold=0.70)
        profiles = {
            (SCHEMA, "patient", "patient_id"): (10, 30),
            (SCHEMA, "sample", "patient_id"): (5, 30),
        }
        out = det.detect(
            columns=self.cols, source_schema=SCHEMA, profiles=profiles,
        )
        assert out == []


class TestContainmentEvidence:
    PK_KEY = (SCHEMA, "patient", "patient_id")
    FK_KEY = (SCHEMA, "sample", "patient_id")

    def setup_method(self):
        self.cols = [
            _col("patient_id", "patient"),
            _col("patient_id", "sample"),
        ]

    def test_only_the_referencing_column_is_sampled(self):
        sampler_calls: list[tuple] = []
        containment_calls: list[tuple] = []

        def sampler(key):
            sampler_calls.append(key)
            return {"a", "b"}

        def containment(key, values):
            containment_calls.append((key, values))
            return 2

        out = JoinDetector().detect(
            columns=self.cols, source_schema=SCHEMA,
            sampler=sampler, containment=containment,
        )
        assert sampler_calls == [self.FK_KEY]
        assert containment_calls == [(self.PK_KEY, frozenset({"a", "b"}))]
        assert out[0].tier == 1

    def test_containment_values_are_bounded_by_sample_cap(self):
        seen: list[frozenset[str]] = []

        def containment(key, values):
            seen.append(values)
            return len(values)

        JoinDetector(sample_cap=3).detect(
            columns=self.cols, source_schema=SCHEMA,
            sampler=lambda key: {"a", "b", "c", "d", "e"},
            containment=containment,
        )
        assert [len(values) for values in seen] == [3]

    def test_containment_failure_falls_through_to_tier_2(self):
        profiles = {
            self.PK_KEY: (100, 100),
            self.FK_KEY: (90, 200),
        }
        out = JoinDetector().detect(
            columns=self.cols, source_schema=SCHEMA, profiles=profiles,
            sampler=lambda key: {"a"},
            containment=lambda key, values: None,
        )
        assert out[0].tier == 2

    def test_warehouse_error_falls_through_to_tier_3(self):
        def sampler(key):
            raise RuntimeError("warehouse unreachable")

        out = JoinDetector().detect(
            columns=self.cols, source_schema=SCHEMA,
            sampler=sampler, containment=lambda key, values: 1,
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
        assert assertion.subject_ref == cols[1].ref
        assert assertion.object_ref == cols[0].ref


class TestReferencedColumnWithoutProfile:
    COLS = [_col("item_id", "item"), _col("item_id", "line")]

    def test_containment_cannot_lift_it_above_tier_3(self):
        out = JoinDetector().detect(
            columns=self.COLS, source_schema=SCHEMA, profiles={},
            sampler=lambda key: {"1", "2"},
            containment=lambda key, values: len(values),
        )
        assert [(fk.tier, fk.confidence) for fk in out] == [(3, TIER_3)]

    def test_it_stays_behind_the_structural_opt_in(self):
        out = JoinDetector().detect(
            columns=self.COLS, source_schema=SCHEMA,
            sampler=lambda key: {"1", "2"},
            containment=lambda key, values: len(values),
        )
        assert JoinDetector().should_materialize(out[0]) is False
        assert JoinDetector(materialization_threshold=0.70).should_materialize(out[0]) is True
