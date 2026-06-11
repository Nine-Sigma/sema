"""US-003: order-independent shared-node writes (confidence-guarded SET).

Description/source/confidence on :Entity/:Property/:Term must update only when
the incoming confidence is strictly greater than the stored confidence, so
build order cannot decide what a node shared across studies says (finding C).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sema.graph.loader_utils import (
    UNSET_CONFIDENCE,
    batch_upsert_entities,
    batch_upsert_properties,
    batch_upsert_terms,
    confidence_wins,
)

pytestmark = pytest.mark.unit


def _loader() -> MagicMock:
    loader = MagicMock()
    loader._run = MagicMock()
    return loader


class TestConfidenceWins:
    def test_high_after_low_overwrites(self):
        assert confidence_wins(0.9, 0.5) is True

    def test_low_after_high_does_not_overwrite(self):
        assert confidence_wins(0.5, 0.9) is False

    def test_equal_keeps_existing(self):
        assert confidence_wins(0.9, 0.9) is False

    def test_missing_stored_writes(self):
        assert confidence_wins(0.5, None) is True

    def test_sentinel_below_any_valid_confidence(self):
        assert UNSET_CONFIDENCE < 0.0


def _apply(stored: float | None, incoming: float) -> float:
    """Simulate the guarded SET: stored value updated only if incoming wins."""
    return incoming if confidence_wins(incoming, stored) else stored  # type: ignore[return-value]


class TestOrderIndependence:
    def test_final_state_identical_regardless_of_order(self):
        writes = [0.5, 0.9, 0.7]
        forward: float | None = None
        for c in writes:
            forward = _apply(forward, c)
        backward: float | None = None
        for c in reversed(writes):
            backward = _apply(backward, c)
        assert forward == backward == 0.9

    def test_low_then_high_equals_high_then_low(self):
        assert _apply(_apply(None, 0.3), 0.9) == _apply(_apply(None, 0.9), 0.3)


class TestGuardedCypher:
    def test_entities_guard_description_and_confidence(self):
        loader = _loader()
        batch_upsert_entities(
            loader,
            [{"name": "Patient", "description": "d", "source": "llm",
              "confidence": 0.9, "table_name": "t", "schema_name": "s",
              "catalog": "c"}],
        )
        cypher = loader._run.call_args[0][0]
        assert "ON CREATE SET e.id = r.id" in cypher
        assert f"coalesce(e.confidence, {UNSET_CONFIDENCE})" in cypher
        assert "e.description = CASE WHEN win THEN r.description" in cypher
        assert "e.confidence = CASE WHEN win THEN r.confidence" in cypher

    def test_properties_guard_semantic_type_and_confidence(self):
        loader = _loader()
        batch_upsert_properties(
            loader,
            [{"name": "age", "entity_name": "Patient", "semantic_type": "int",
              "source": "llm", "confidence": 0.9, "column_name": "age",
              "table_name": "t", "schema_name": "s", "catalog": "c"}],
        )
        cypher = loader._run.call_args[0][0]
        assert f"coalesce(p.confidence, {UNSET_CONFIDENCE})" in cypher
        assert "p.semantic_type = CASE WHEN win THEN r.semantic_type" in cypher
        assert "p.confidence = CASE WHEN win THEN r.confidence" in cypher

    def test_terms_guard_label_and_confidence(self):
        loader = _loader()
        batch_upsert_terms(
            loader,
            [{"code": "M", "label": "Male", "vocabulary_name": "Gender",
              "source": "llm", "confidence": 0.9}],
        )
        cypher = loader._run.call_args[0][0]
        assert f"coalesce(t.confidence, {UNSET_CONFIDENCE})" in cypher
        assert "t.label = CASE WHEN win THEN r.label" in cypher
        assert "t.confidence = CASE WHEN win THEN r.confidence" in cypher

    def test_source_id_coalesce_preserved(self):
        loader = _loader()
        batch_upsert_entities(
            loader,
            [{"name": "Patient", "description": "d", "source": "llm",
              "confidence": 0.9, "table_name": "t", "schema_name": "s",
              "catalog": "c"}],
        )
        cypher = loader._run.call_args[0][0]
        assert (
            "e.source_id = coalesce(e.source_id, r.source_schema, r.source)"
            in cypher
        )
