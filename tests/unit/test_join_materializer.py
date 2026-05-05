"""Tests for `sema.graph.join_materializer` source_schema threading."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from sema.engine.join_detector import FKAssertion, to_fk_assertion
from sema.engine.join_detector_utils import FKCandidate
from sema.graph.join_materializer import (
    _build_join_path_records,
    _derive_join_path_name,
    materialize_join_paths,
)
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)

pytestmark = pytest.mark.unit

SCHEMA = "cbioportal_msk_chord_2024"


def _join_assertion(name: str = "patient_to_sample"):
    payload = {
        "join_predicates": [{
            "left_table": "patient", "left_column": "patient_id",
            "right_table": "sample", "right_column": "patient_id",
            "operator": "=",
        }],
        "hop_count": 1,
        "from_table": "databricks://ws/cat/sch/patient",
        "to_table": "databricks://ws/cat/sch/sample",
    }
    return Assertion(
        id=f"a-{name}",
        subject_ref="databricks://ws/cat/sch/patient",
        predicate=AssertionPredicate.HAS_JOIN_EVIDENCE,
        payload=payload,
        source="fk_detector",
        confidence=0.95,
        status=AssertionStatus.AUTO,
        run_id="run-1",
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_derive_join_path_name():
    name = _derive_join_path_name([
        {"left_table": "p", "left_column": "id",
         "right_table": "s", "right_column": "p_id"},
    ])
    assert name == "p/id=s/p_id"


def test_build_join_path_records_skips_empty_group():
    records = _build_join_path_records({"x": []})
    assert records == []


def test_materialize_threads_source_schema_to_batch():
    loader = MagicMock()
    a = _join_assertion()
    groups = {(a.subject_ref, a.predicate.value): [a]}
    materialize_join_paths(loader, groups, source_schema=SCHEMA)
    loader.add_join_path_uses.assert_called()
    for call in loader.add_join_path_uses.call_args_list:
        assert call.kwargs.get("source_schema") == SCHEMA
    if loader.add_join_path_entity_links.called:
        for call in loader.add_join_path_entity_links.call_args_list:
            assert call.kwargs.get("source_schema") == SCHEMA


def test_materialize_skips_edge_writes_when_source_schema_missing():
    """Edge writes require source_schema; skip if absent (legacy)."""
    loader = MagicMock()
    a = _join_assertion()
    groups = {(a.subject_ref, a.predicate.value): [a]}
    materialize_join_paths(loader, groups)
    loader.add_join_path_uses.assert_not_called()
    loader.add_join_path_entity_links.assert_not_called()


def _fk_to_assertion(tier: int = 1, confidence: float = 0.95) -> Assertion:
    candidate = FKCandidate(
        pk_table="patient", pk_column="patient_id",
        fk_table="sample", fk_column="patient_id",
        pk_type="string", fk_type="string",
        schema_name=SCHEMA, catalog="workspace",
    )
    fk = FKAssertion(
        candidate=candidate, confidence=confidence,
        tier=tier, source_schema=SCHEMA,
    )
    return to_fk_assertion(fk, run_id="run-1")


def test_fk_to_predicate_produces_join_path():
    loader = MagicMock()
    a = _fk_to_assertion()
    groups = {(a.subject_ref, a.predicate.value): [a]}
    materialize_join_paths(loader, groups, source_schema=SCHEMA)
    batch_calls = [
        c for c in loader.method_calls if c[0] == "_run"
    ]
    assert batch_calls, "expected JoinPath upsert to fire"
    rows = batch_calls[0].kwargs["rows"]
    assert len(rows) == 1
    rec = rows[0]
    assert rec["name"] == "patient/patient_id=sample/patient_id"
    assert rec["confidence"] == 0.95
    assert rec["source_schema"] == SCHEMA
    loader.add_join_path_uses.assert_called()
    loader.add_join_path_entity_links.assert_called()
    for call in loader.add_join_path_uses.call_args_list:
        assert call.kwargs.get("source_schema") == SCHEMA


def test_fk_to_and_legacy_join_evidence_coexist():
    loader = MagicMock()
    legacy = _join_assertion(name="legacy")
    fk = _fk_to_assertion()
    groups = {
        (legacy.subject_ref, legacy.predicate.value): [legacy],
        (fk.subject_ref, fk.predicate.value): [fk],
    }
    materialize_join_paths(loader, groups, source_schema=SCHEMA)
    rows = [
        c for c in loader.method_calls if c[0] == "_run"
    ][0].kwargs["rows"]
    names = sorted(r["name"] for r in rows)
    assert names == [
        "patient/patient_id=sample/patient_id",
        "patient/patient_id=sample/patient_id",
    ]
    assert len(rows) == 2
