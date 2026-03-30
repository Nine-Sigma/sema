"""Tests for assertion family key stability and collision avoidance."""

import pytest

from sema.models.assertions import AssertionPredicate
from sema.models.family_key import family_key, payload_identity


pytestmark = pytest.mark.unit


class TestPayloadIdentity:
    def test_vocabulary_match_extracts_value(self) -> None:
        pid = payload_identity(
            AssertionPredicate.VOCABULARY_MATCH, {"value": "ICD-10"}
        )
        assert pid == "ICD-10"

    def test_decoded_value_extracts_code(self) -> None:
        pid = payload_identity(
            AssertionPredicate.HAS_DECODED_VALUE,
            {"code": "C34.1", "label": "Lung cancer"},
        )
        assert pid == "C34.1"

    def test_parent_of_returns_none(self) -> None:
        pid = payload_identity(
            AssertionPredicate.PARENT_OF,
            {"parent": "C34", "child": "C34.1"},
        )
        assert pid is None

    def test_table_exists_returns_none(self) -> None:
        pid = payload_identity(
            AssertionPredicate.TABLE_EXISTS, {}
        )
        assert pid is None


class TestFamilyKey:
    def test_same_vocab_match_across_runs_same_key(self) -> None:
        k1 = family_key(
            "col_X",
            AssertionPredicate.VOCABULARY_MATCH,
            {"value": "ICD-10"},
        )
        k2 = family_key(
            "col_X",
            AssertionPredicate.VOCABULARY_MATCH,
            {"value": "ICD-10"},
        )
        assert k1 == k2

    def test_different_vocab_value_different_key(self) -> None:
        k1 = family_key(
            "col_X",
            AssertionPredicate.VOCABULARY_MATCH,
            {"value": "ICD-10"},
        )
        k2 = family_key(
            "col_X",
            AssertionPredicate.VOCABULARY_MATCH,
            {"value": "CPT"},
        )
        assert k1 != k2

    def test_parent_of_same_subject_object_same_key(self) -> None:
        k1 = family_key(
            "col_X",
            AssertionPredicate.PARENT_OF,
            {"parent": "C34", "child": "C34.1"},
            object_ref="term:C34.1",
        )
        k2 = family_key(
            "col_X",
            AssertionPredicate.PARENT_OF,
            {"parent": "C34", "child": "C34.1"},
            object_ref="term:C34.1",
        )
        assert k1 == k2

    def test_parent_of_ignores_confidence_in_payload(self) -> None:
        k1 = family_key(
            "col_X",
            AssertionPredicate.PARENT_OF,
            {"parent": "C34", "child": "C34.1", "confidence": 0.8},
            object_ref="term:C34.1",
        )
        k2 = family_key(
            "col_X",
            AssertionPredicate.PARENT_OF,
            {"parent": "C34", "child": "C34.1", "confidence": 0.95},
            object_ref="term:C34.1",
        )
        assert k1 == k2

    def test_parent_of_swapped_subject_object_different_key(self) -> None:
        k1 = family_key(
            "col_X",
            AssertionPredicate.PARENT_OF,
            {},
            object_ref="term:C34.1",
        )
        k2 = family_key(
            "term:C34.1",
            AssertionPredicate.PARENT_OF,
            {},
            object_ref="col_X",
        )
        assert k1 != k2

    def test_different_subject_ref_different_key(self) -> None:
        k1 = family_key(
            "col_A",
            AssertionPredicate.VOCABULARY_MATCH,
            {"value": "ICD-10"},
        )
        k2 = family_key(
            "col_B",
            AssertionPredicate.VOCABULARY_MATCH,
            {"value": "ICD-10"},
        )
        assert k1 != k2
