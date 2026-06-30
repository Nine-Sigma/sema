"""US-002: :Term identity keyed on {vocabulary_name, code}.

Unrelated codes in different vocabularies (M=Male vs M=Mississippi) must
never collapse into one node. Vocabulary-less terms use the UNSCOPED_VOCAB
sentinel so the namespace component of the key is never null.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sema.graph.loader import GraphLoader
from sema.graph.loader_utils import (
    batch_create_in_vocabulary,
    batch_upsert_terms,
)
from sema.graph.materializer_utils import upsert_decoded_values
from sema.graph.term_identity_utils import UNSCOPED_VOCAB, term_namespace
from sema.graph.term_vocab_utils import resolve_term_vocab, term_vocab_for_subject
from sema.models.assertions import AssertionPredicate, AssertionStatus
from tests.conftest import make_assertion

pytestmark = pytest.mark.unit

REF_COL = "databricks://ws/cdm/clinical/patients/sex"


def _loader_with_run() -> MagicMock:
    loader = MagicMock()
    loader._run = MagicMock()
    return loader


@pytest.fixture
def driver_loader():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    loader = GraphLoader.__new__(GraphLoader)
    loader._driver = driver
    return loader, session


class TestTermNamespace:
    def test_none_uses_sentinel(self):
        assert term_namespace(None) == UNSCOPED_VOCAB

    def test_empty_uses_sentinel(self):
        assert term_namespace("") == UNSCOPED_VOCAB

    def test_real_name_preserved(self):
        assert term_namespace("Gender") == "Gender"


class TestBatchUpsertTermsCompositeKey:
    def test_merges_on_vocabulary_and_code(self):
        loader = _loader_with_run()
        batch_upsert_terms(
            loader,
            [{"code": "M", "label": "Male", "vocabulary_name": "Gender",
              "source": "llm", "confidence": 0.9}],
        )
        cypher = loader._run.call_args[0][0]
        assert (
            "MERGE (t:Term {vocabulary_name: r.vocabulary_name, code: r.code})"
            in cypher
        )

    def test_vocabulary_less_term_uses_sentinel(self):
        loader = _loader_with_run()
        batch_upsert_terms(
            loader,
            [{"code": "M", "label": "Male", "source": "llm",
              "confidence": 0.9}],
        )
        rows = loader._run.call_args[1]["rows"]
        assert rows[0]["vocabulary_name"] == UNSCOPED_VOCAB

    def test_same_code_two_vocabs_two_distinct_keys(self):
        loader = _loader_with_run()
        batch_upsert_terms(
            loader,
            [
                {"code": "M", "label": "Male", "vocabulary_name": "Gender",
                 "source": "llm", "confidence": 0.9},
                {"code": "M", "label": "Mississippi",
                 "vocabulary_name": "State", "source": "llm",
                 "confidence": 0.9},
            ],
        )
        rows = loader._run.call_args[1]["rows"]
        keys = {(r["vocabulary_name"], r["code"]) for r in rows}
        assert keys == {("Gender", "M"), ("State", "M")}

    def test_same_code_same_vocab_one_key(self):
        loader = _loader_with_run()
        batch_upsert_terms(
            loader,
            [
                {"code": "M", "label": "Male", "vocabulary_name": "Gender",
                 "source": "llm", "confidence": 0.9},
                {"code": "M", "label": "Male", "vocabulary_name": "Gender",
                 "source": "llm", "confidence": 0.9},
            ],
        )
        rows = loader._run.call_args[1]["rows"]
        keys = {(r["vocabulary_name"], r["code"]) for r in rows}
        assert keys == {("Gender", "M")}


class TestInVocabularyCompositeMatch:
    def test_matches_term_on_composite_key(self):
        loader = _loader_with_run()
        batch_create_in_vocabulary(
            loader, [{"vocabulary_name": "Gender", "code": "M"}],
        )
        cypher = loader._run.call_args[0][0]
        assert (
            "MATCH (t:Term {vocabulary_name: r.vocabulary_name, code: r.code})"
            in cypher
        )


class TestSingleRowTermMethods:
    def test_upsert_term_composite_key_with_sentinel(self, driver_loader):
        loader, session = driver_loader
        loader.upsert_term("M", "Male", source="llm", confidence=0.9)
        cypher = session.run.call_args[0][0]
        params = session.run.call_args[1]
        assert (
            "MERGE (t:Term {vocabulary_name: $vocabulary_name, code: $code})"
            in cypher
        )
        assert params["vocabulary_name"] == UNSCOPED_VOCAB

    def test_upsert_term_uses_supplied_vocabulary(self, driver_loader):
        loader, session = driver_loader
        loader.upsert_term(
            "M", "Male", source="llm", confidence=0.9,
            vocabulary_name="Gender",
        )
        params = session.run.call_args[1]
        assert params["vocabulary_name"] == "Gender"

    def test_add_term_to_value_set_composite_key(self, driver_loader):
        loader, session = driver_loader
        loader.add_term_to_value_set(
            "M", "sex_values", source_schema="sch",
            vocabulary_name="sex_values",
        )
        cypher = session.run.call_args[0][0]
        params = session.run.call_args[1]
        assert (
            "MERGE (t:Term {vocabulary_name: $vocabulary_name, "
            "code: $term_code})" in cypher
        )
        assert params["vocabulary_name"] == "sex_values"

    def test_add_term_hierarchy_composite_key(self, driver_loader):
        loader, session = driver_loader
        loader.add_term_hierarchy(
            "NEOPLASM", "CARCINOMA", source_schema="sch",
            vocabulary_name="ICD",
        )
        cypher = session.run.call_args[0][0]
        params = session.run.call_args[1]
        assert (
            "MERGE (p:Term {vocabulary_name: $vocabulary_name, "
            "code: $parent_code})" in cypher
        )
        assert (
            "MERGE (c:Term {vocabulary_name: $vocabulary_name, "
            "code: $child_code})" in cypher
        )
        assert params["vocabulary_name"] == "ICD"


class TestResolveTermVocab:
    def test_returns_matched_vocabulary(self):
        groups = {
            (REF_COL, AssertionPredicate.VOCABULARY_MATCH.value): [
                make_assertion(
                    subject_ref=REF_COL,
                    predicate=AssertionPredicate.VOCABULARY_MATCH.value,
                    value={"value": "Gender"},
                ),
            ],
        }
        assert resolve_term_vocab(REF_COL, groups, "fallback") == "Gender"

    def test_falls_back_when_no_match(self):
        assert resolve_term_vocab(REF_COL, {}, "sex_values") == "sex_values"

    def test_subject_helper_uses_value_set_name(self):
        assert term_vocab_for_subject(REF_COL, {}) == "patients_sex_values"

    def test_subject_helper_bad_ref_uses_sentinel(self):
        assert term_vocab_for_subject("not-a-ref", {}) == UNSCOPED_VOCAB


class TestDecodedValuesNamespacing:
    def test_terms_namespaced_by_matched_vocabulary(self):
        loader = MagicMock()
        groups = {
            (REF_COL, AssertionPredicate.HAS_DECODED_VALUE.value): [
                make_assertion(
                    subject_ref=REF_COL,
                    predicate=AssertionPredicate.HAS_DECODED_VALUE.value,
                    value={"raw": "M", "label": "Male"},
                    status=AssertionStatus.AUTO,
                ),
            ],
            (REF_COL, AssertionPredicate.VOCABULARY_MATCH.value): [
                make_assertion(
                    subject_ref=REF_COL,
                    predicate=AssertionPredicate.VOCABULARY_MATCH.value,
                    value={"value": "Gender"},
                ),
            ],
        }
        captured: dict[str, object] = {}

        def capture(_loader, terms, source_schema=None):
            captured["terms"] = terms

        import sema.graph.materializer_utils as mu

        original = mu.batch_upsert_terms
        mu.batch_upsert_terms = capture
        try:
            upsert_decoded_values(loader, groups, source_schema="sch")
        finally:
            mu.batch_upsert_terms = original

        terms = captured["terms"]
        assert terms[0]["vocabulary_name"] == "Gender"
        loader.add_term_to_value_set.assert_called_once()
        assert loader.add_term_to_value_set.call_args[1][
            "vocabulary_name"
        ] == "Gender"

    def test_terms_fall_back_to_value_set_name(self):
        loader = MagicMock()
        groups = {
            (REF_COL, AssertionPredicate.HAS_DECODED_VALUE.value): [
                make_assertion(
                    subject_ref=REF_COL,
                    predicate=AssertionPredicate.HAS_DECODED_VALUE.value,
                    value={"raw": "M", "label": "Male"},
                    status=AssertionStatus.AUTO,
                ),
            ],
        }
        captured: dict[str, object] = {}

        def capture(_loader, terms, source_schema=None):
            captured["terms"] = terms

        import sema.graph.materializer_utils as mu

        original = mu.batch_upsert_terms
        mu.batch_upsert_terms = capture
        try:
            upsert_decoded_values(loader, groups, source_schema="sch")
        finally:
            mu.batch_upsert_terms = original

        terms = captured["terms"]
        assert terms[0]["vocabulary_name"] == "patients_sex_values"
