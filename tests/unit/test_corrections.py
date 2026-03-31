"""Tests for human feedback loop: corrections as assertions."""

import pytest
from datetime import datetime, timezone
from pathlib import Path

from sema.engine.corrections import (
    confirm_assertion,
    reject_assertion,
    relabel_assertion,
    replay_corrections,
)
from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.family_key import family_key
from sema.models.lifecycle import AssertionStatusValue
from sema.models.status_store import StatusEventStore, effective_status

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path: Path) -> StatusEventStore:
    return StatusEventStore(tmp_path / "events")


def _assertion(
    id: str = "a1",
    predicate: AssertionPredicate = AssertionPredicate.VOCABULARY_MATCH,
    payload: dict | None = None,
) -> Assertion:
    return Assertion(
        id=id,
        subject_ref="col_X",
        predicate=predicate,
        payload=payload or {"value": "CPT"},
        source="pattern_match",
        confidence=0.9,
        run_id="run-1",
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


class TestConfirmReject:
    def test_confirm_pins(self, store: StatusEventStore) -> None:
        event = confirm_assertion(store, "ds1", "a1")
        assert event.status == AssertionStatusValue.PINNED
        assert effective_status(store, "ds1", "a1") == AssertionStatusValue.PINNED

    def test_reject_excludes(self, store: StatusEventStore) -> None:
        event = reject_assertion(store, "ds1", "a1", reason="wrong vocab")
        assert event.status == AssertionStatusValue.REJECTED
        assert event.reason == "wrong vocab"
        assert effective_status(store, "ds1", "a1") == AssertionStatusValue.REJECTED


class TestRelabel:
    def test_relabel_produces_three_artifacts(
        self, store: StatusEventStore,
    ) -> None:
        reject_event, new_assertion, pin_event = relabel_assertion(
            store, "ds1", "a1",
            new_subject_ref="col_X",
            new_predicate=AssertionPredicate.VOCABULARY_MATCH,
            new_payload={"value": "ZIP"},
            run_id="run-1",
            reason="column contains ZIP codes",
        )
        # Old assertion rejected
        assert reject_event.status == AssertionStatusValue.REJECTED
        assert effective_status(store, "ds1", "a1") == AssertionStatusValue.REJECTED

        # New assertion created with human source
        assert new_assertion.source == "human"
        assert new_assertion.payload["value"] == "ZIP"
        assert new_assertion.confidence == 1.0

        # New assertion pinned
        assert pin_event.status == AssertionStatusValue.PINNED
        assert effective_status(store, "ds1", new_assertion.id) == AssertionStatusValue.PINNED


class TestCorrectionReplay:
    def test_correction_replayed_by_family_key(
        self, store: StatusEventStore,
    ) -> None:
        """A stored correction for CPT on col_X should apply to a new
        assertion with the same family key in a new run."""
        old = _assertion(id="old_a1")
        fk = family_key(old.subject_ref, old.predicate, old.payload)

        # Store the correction
        corrections = {fk: AssertionStatusValue.REJECTED}

        # New run produces same family
        new = _assertion(id="new_a1")
        events = replay_corrections(store, "ds1", [new], corrections)

        assert len(events) == 1
        assert events[0].assertion_id == "new_a1"
        assert events[0].status == AssertionStatusValue.REJECTED
        assert effective_status(store, "ds1", "new_a1") == AssertionStatusValue.REJECTED

    def test_different_family_not_replayed(
        self, store: StatusEventStore,
    ) -> None:
        """Correction for CPT should NOT apply to ZIP assertion."""
        old = _assertion(id="old_a1", payload={"value": "CPT"})
        fk = family_key(old.subject_ref, old.predicate, old.payload)
        corrections = {fk: AssertionStatusValue.REJECTED}

        # New assertion has different value -> different family
        new = _assertion(id="new_a1", payload={"value": "ZIP"})
        events = replay_corrections(store, "ds1", [new], corrections)

        assert len(events) == 0
        assert effective_status(store, "ds1", "new_a1") == AssertionStatusValue.AUTO
