"""Unit tests for sema.graph.vocabulary_materializer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_assertion
from sema.models.assertions import AssertionPredicate, AssertionStatus

pytestmark = pytest.mark.unit

from sema.graph.vocabulary_materializer import (
    _collect_in_vocabulary_edges,
    materialize_vocabulary_edges,
)

REF_COL = "databricks://ws/cdm/clinical/patients/gender"


def _vocab_assertion(value="ICD-10", status=AssertionStatus.AUTO):
    return make_assertion(
        subject_ref=REF_COL,
        predicate=AssertionPredicate.VOCABULARY_MATCH.value,
        value=value,
        source="pattern_match",
        status=status,
    )


def _decoded_assertion(raw="M", label="Male", status=AssertionStatus.AUTO):
    return make_assertion(
        subject_ref=REF_COL,
        predicate=AssertionPredicate.HAS_DECODED_VALUE.value,
        value={"raw": raw, "label": label},
        source="llm_interpretation",
        status=status,
    )


class TestMaterializeVocabularyEdges:
    def test_creates_vocabulary_and_classified_edges(self):
        loader = MagicMock()
        groups = {
            (REF_COL, AssertionPredicate.VOCABULARY_MATCH.value): [
                _vocab_assertion("ICD-10"),
            ],
        }
        mock_upsert_vocabs = MagicMock()
        mock_classified = MagicMock()
        mock_in_vocab = MagicMock()
        with patch(
            "sema.graph.vocabulary_materializer.batch_upsert_vocabularies",
            mock_upsert_vocabs,
        ), patch(
            "sema.graph.vocabulary_materializer.batch_create_classified_as",
            mock_classified,
        ), patch(
            "sema.graph.vocabulary_materializer.batch_create_in_vocabulary",
            mock_in_vocab,
        ):
            materialize_vocabulary_edges(loader, groups)

        mock_upsert_vocabs.assert_called_once()
        vocab_batch = mock_upsert_vocabs.call_args[0][1]
        assert any(v["name"] == "ICD-10" for v in vocab_batch)

        mock_classified.assert_called_once()
        edges = mock_classified.call_args[0][1]
        assert len(edges) == 1
        assert edges[0]["vocabulary_name"] == "ICD-10"

    def test_skips_rejected_vocab_match(self):
        loader = MagicMock()
        groups = {
            (REF_COL, AssertionPredicate.VOCABULARY_MATCH.value): [
                _vocab_assertion(status=AssertionStatus.REJECTED),
            ],
        }
        mock_upsert = MagicMock()
        mock_classified = MagicMock()
        with patch(
            "sema.graph.vocabulary_materializer.batch_upsert_vocabularies",
            mock_upsert,
        ), patch(
            "sema.graph.vocabulary_materializer.batch_create_classified_as",
            mock_classified,
        ), patch(
            "sema.graph.vocabulary_materializer.batch_create_in_vocabulary",
            MagicMock(),
        ):
            materialize_vocabulary_edges(loader, groups)

        mock_upsert.assert_not_called()
        edges = mock_classified.call_args[0][1]
        assert len(edges) == 0

    def test_skips_invalid_ref(self):
        loader = MagicMock()
        groups = {
            ("bad-ref", AssertionPredicate.VOCABULARY_MATCH.value): [
                make_assertion(
                    subject_ref="bad-ref",
                    predicate=AssertionPredicate.VOCABULARY_MATCH.value,
                    value="ICD-10",
                ),
            ],
        }
        mock_classified = MagicMock()
        with patch(
            "sema.graph.vocabulary_materializer.batch_upsert_vocabularies",
            MagicMock(),
        ), patch(
            "sema.graph.vocabulary_materializer.batch_create_classified_as",
            mock_classified,
        ), patch(
            "sema.graph.vocabulary_materializer.batch_create_in_vocabulary",
            MagicMock(),
        ):
            materialize_vocabulary_edges(loader, groups)

        edges = mock_classified.call_args[0][1]
        assert len(edges) == 0


class TestCollectInVocabularyEdges:
    def test_collects_edges_for_decoded_values(self):
        groups = {
            (REF_COL, AssertionPredicate.HAS_DECODED_VALUE.value): [
                _decoded_assertion("M", "Male"),
                _decoded_assertion("F", "Female"),
            ],
            (REF_COL, AssertionPredicate.VOCABULARY_MATCH.value): [
                _vocab_assertion("Gender"),
            ],
        }
        edges = _collect_in_vocabulary_edges(groups)
        assert len(edges) == 2
        assert all(e["vocabulary_name"] == "Gender" for e in edges)
        codes = {e["code"] for e in edges}
        assert codes == {"M", "F"}

    def test_skips_when_no_vocab_match(self):
        groups = {
            (REF_COL, AssertionPredicate.HAS_DECODED_VALUE.value): [
                _decoded_assertion("M", "Male"),
            ],
        }
        edges = _collect_in_vocabulary_edges(groups)
        assert len(edges) == 0

    def test_skips_rejected_decoded_values(self):
        groups = {
            (REF_COL, AssertionPredicate.HAS_DECODED_VALUE.value): [
                _decoded_assertion("X", "Unknown", status=AssertionStatus.REJECTED),
            ],
            (REF_COL, AssertionPredicate.VOCABULARY_MATCH.value): [
                _vocab_assertion("Gender"),
            ],
        }
        edges = _collect_in_vocabulary_edges(groups)
        assert len(edges) == 0
