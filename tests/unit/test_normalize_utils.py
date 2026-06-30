"""Tests for name normalization at the assertion boundary (US-001)."""

import unicodedata
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from sema.engine.normalize_utils import normalize_name
from sema.graph.materializer_utils import upsert_entities
from sema.models.assertions import Assertion, AssertionPredicate

pytestmark = pytest.mark.unit


class TestNormalizeName:
    def test_strips_leading_and_trailing_whitespace(self) -> None:
        assert normalize_name("  Patient  ") == "Patient"

    def test_collapses_internal_whitespace_runs(self) -> None:
        assert normalize_name("Tumor   Sample") == "Tumor Sample"
        assert normalize_name("a\t\tb\nc") == "a b c"

    def test_unicode_nfc_equivalence(self) -> None:
        nfd = unicodedata.normalize("NFD", "café")
        nfc = unicodedata.normalize("NFC", "café")
        assert nfd != nfc  # they really are different code point sequences
        assert normalize_name(nfd) == normalize_name(nfc) == nfc

    def test_case_is_preserved(self) -> None:
        assert normalize_name("Patient ID") == "Patient ID"

    def test_idempotent(self) -> None:
        once = normalize_name("  Tumor   Sample  ")
        assert normalize_name(once) == once

    def test_empty_string(self) -> None:
        assert normalize_name("") == ""

    def test_whitespace_only_becomes_empty(self) -> None:
        assert normalize_name("   ") == ""


def _name_assertion(predicate: AssertionPredicate, value: str) -> Assertion:
    return Assertion(
        id=str(uuid.uuid4()),
        subject_ref="databricks://ws/cat/sch/tbl",
        predicate=predicate,
        payload={"value": value},
        source="llm_interpretation",
        confidence=0.75,
        run_id="run-1",
        observed_at=datetime.now(timezone.utc),
    )


class TestAssertionNormalizesNamePayload:
    @pytest.mark.parametrize(
        "predicate",
        [
            AssertionPredicate.HAS_ENTITY_NAME,
            AssertionPredicate.HAS_PROPERTY_NAME,
            AssertionPredicate.HAS_ALIAS,
            AssertionPredicate.HAS_SYNONYM,
        ],
    )
    def test_value_normalized_on_construction(
        self, predicate: AssertionPredicate
    ) -> None:
        a = _name_assertion(predicate, "  Tumor   Sample ")
        assert a.payload["value"] == "Tumor Sample"

    def test_non_name_predicate_value_untouched(self) -> None:
        a = Assertion(
            id="1",
            subject_ref="databricks://ws/cat/sch/tbl",
            predicate=AssertionPredicate.HAS_COMMENT,
            payload={"value": "  keep   spaces "},
            source="llm_interpretation",
            confidence=0.95,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
        )
        assert a.payload["value"] == "  keep   spaces "

    def test_missing_value_key_is_safe(self) -> None:
        a = Assertion(
            id="1",
            subject_ref="databricks://ws/cat/sch/tbl",
            predicate=AssertionPredicate.HAS_ENTITY_NAME,
            payload={},
            source="llm_interpretation",
            confidence=0.75,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
        )
        assert a.payload == {}


class TestWhitespaceVariantsCollapseToOneEntity:
    def test_two_variant_names_produce_one_merge_key(self) -> None:
        plain = _name_assertion(AssertionPredicate.HAS_ENTITY_NAME, "Tumor Sample")
        nfd = unicodedata.normalize("NFD", "Tumor Sample")
        spaced = _name_assertion(
            AssertionPredicate.HAS_ENTITY_NAME, f"  {nfd}   "
        )

        groups = {
            ("databricks://ws/cat/sch/a", AssertionPredicate.HAS_ENTITY_NAME.value): [
                plain
            ],
            ("databricks://ws/cat/sch/b", AssertionPredicate.HAS_ENTITY_NAME.value): [
                spaced
            ],
        }
        captured: list[list[dict[str, object]]] = []
        loader = MagicMock()

        def fake_run(query: str, rows: list[dict[str, object]]) -> object:
            captured.append(rows)
            return MagicMock()

        loader._run.side_effect = fake_run

        upsert_entities(loader, groups, source_schema="sch")

        names = {row["name"] for rows in captured for row in rows}
        assert names == {"Tumor Sample"}
