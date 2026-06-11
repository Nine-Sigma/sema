"""US-007: source_schema on IN_VOCABULARY and CLASSIFIED_AS edges.

These two vocabulary-association edge types previously carried no
``source_schema``, so they survived ``delete_study_scoped`` (which sweeps
edges by that property). Keying their MERGE on ``source_schema`` — mirroring
HAS_VALUE_SET / MEMBER_OF — makes them per-study and independently deletable.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sema.graph.loader_utils import (
    batch_create_classified_as,
    batch_create_in_vocabulary,
)
from sema.graph.vocabulary_materializer import materialize_vocabulary_edges
from sema.models.assertions import AssertionPredicate, AssertionStatus
from tests.conftest import make_assertion

pytestmark = pytest.mark.unit

REF_COL = "databricks://ws/cdm/clinical/patients/gender"


def _loader_with_run() -> MagicMock:
    loader = MagicMock()
    loader._run = MagicMock()
    return loader


class TestClassifiedAsScoping:
    def test_merge_key_includes_source_schema(self):
        loader = _loader_with_run()
        batch_create_classified_as(
            loader,
            [{"entity_name": "Patient", "name": "gender",
              "vocabulary_name": "ICD-10", "source_schema": "study_a"}],
        )
        cypher = loader._run.call_args[0][0]
        assert (
            "MERGE (p)-[:CLASSIFIED_AS "
            "{source_schema: r.source_schema}]->(v)" in cypher
        )

    def test_two_studies_produce_two_edge_rows(self):
        loader = _loader_with_run()
        edges = [
            {"entity_name": "Patient", "name": "gender",
             "vocabulary_name": "ICD-10", "source_schema": "study_a"},
            {"entity_name": "Patient", "name": "gender",
             "vocabulary_name": "ICD-10", "source_schema": "study_b"},
        ]
        batch_create_classified_as(loader, edges)
        rows = loader._run.call_args[1]["rows"]
        schemas = {r["source_schema"] for r in rows}
        assert schemas == {"study_a", "study_b"}


class TestInVocabularyScoping:
    def test_merge_key_includes_source_schema(self):
        loader = _loader_with_run()
        batch_create_in_vocabulary(
            loader,
            [{"vocabulary_name": "Gender", "code": "M",
              "source_schema": "study_a"}],
        )
        cypher = loader._run.call_args[0][0]
        assert (
            "MERGE (t)-[:IN_VOCABULARY "
            "{source_schema: r.source_schema}]->(v)" in cypher
        )

    def test_two_studies_produce_two_edge_rows(self):
        loader = _loader_with_run()
        edges = [
            {"vocabulary_name": "Gender", "code": "M",
             "source_schema": "study_a"},
            {"vocabulary_name": "Gender", "code": "M",
             "source_schema": "study_b"},
        ]
        batch_create_in_vocabulary(loader, edges)
        rows = loader._run.call_args[1]["rows"]
        schemas = {r["source_schema"] for r in rows}
        assert schemas == {"study_a", "study_b"}


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


class TestMaterializeThreadsSourceSchema:
    def test_classified_and_in_vocab_edges_carry_source_schema(self):
        loader = MagicMock()
        groups = {
            (REF_COL, AssertionPredicate.VOCABULARY_MATCH.value): [
                _vocab_assertion("ICD-10"),
            ],
            (REF_COL, AssertionPredicate.HAS_DECODED_VALUE.value): [
                _decoded_assertion("M", "Male"),
            ],
        }
        mock_classified = MagicMock()
        mock_in_vocab = MagicMock()
        with patch(
            "sema.graph.vocabulary_materializer.batch_upsert_vocabularies",
            MagicMock(),
        ), patch(
            "sema.graph.vocabulary_materializer.batch_create_classified_as",
            mock_classified,
        ), patch(
            "sema.graph.vocabulary_materializer.batch_create_in_vocabulary",
            mock_in_vocab,
        ):
            materialize_vocabulary_edges(
                loader, groups, source_schema="study_a",
            )

        classified_edges = mock_classified.call_args[0][1]
        assert classified_edges
        assert all(
            e["source_schema"] == "study_a" for e in classified_edges
        )

        in_vocab_edges = mock_in_vocab.call_args[0][1]
        assert in_vocab_edges
        assert all(
            e["source_schema"] == "study_a" for e in in_vocab_edges
        )
