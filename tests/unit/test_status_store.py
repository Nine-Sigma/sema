"""Tests for StatusEvent store and effective_status."""

import pytest
from datetime import datetime, timezone
from pathlib import Path

from sema.models.lifecycle import AssertionStatusValue, StatusEvent
from sema.models.status_store import StatusEventStore, effective_status

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path: Path) -> StatusEventStore:
    return StatusEventStore(tmp_path / "status_events")


def _event(
    assertion_id: str,
    status: AssertionStatusValue,
    actor: str = "machine",
) -> StatusEvent:
    return StatusEvent(
        assertion_id=assertion_id,
        status=status,
        actor=actor,
        timestamp=datetime.now(timezone.utc),
    )


class TestStatusEventStore:
    def test_append_and_query(self, store: StatusEventStore) -> None:
        event = _event("a1", AssertionStatusValue.PINNED, "human")
        store.append("ds1", event)
        results = store.query_by_assertion_id("ds1", "a1")
        assert len(results) == 1
        assert results[0].status == AssertionStatusValue.PINNED

    def test_query_latest(self, store: StatusEventStore) -> None:
        store.append("ds1", _event("a1", AssertionStatusValue.AUTO))
        store.append("ds1", _event("a1", AssertionStatusValue.PINNED))
        latest = store.query_latest_by_assertion_id("ds1", "a1")
        assert latest is not None
        assert latest.status == AssertionStatusValue.PINNED

    def test_query_latest_returns_none_for_unknown(
        self, store: StatusEventStore
    ) -> None:
        result = store.query_latest_by_assertion_id("ds1", "nonexistent")
        assert result is None

    def test_bulk_load(self, store: StatusEventStore) -> None:
        events = [
            _event("a1", AssertionStatusValue.AUTO),
            _event("a2", AssertionStatusValue.REJECTED),
            _event("a3", AssertionStatusValue.PINNED),
        ]
        store.bulk_load("ds1", events)
        assert len(store.query_by_assertion_id("ds1", "a1")) == 1
        assert len(store.query_by_assertion_id("ds1", "a2")) == 1
        assert len(store.query_by_assertion_id("ds1", "a3")) == 1

    def test_empty_bulk_load(self, store: StatusEventStore) -> None:
        store.bulk_load("ds1", [])
        # Should not create file
        assert store.query_by_assertion_id("ds1", "a1") == []

    def test_query_by_family_key_no_index(
        self, store: StatusEventStore
    ) -> None:
        assert store.query_by_family_key("ds1", "fk1") == []
        assert store.query_by_family_key("ds1", "fk1", None) == []

    def test_query_by_family_key_missing_key(
        self, store: StatusEventStore
    ) -> None:
        assert store.query_by_family_key("ds1", "fk1", {"other": "a1"}) == []

    def test_query_by_family_key_found(
        self, store: StatusEventStore
    ) -> None:
        store.append("ds1", _event("a1", AssertionStatusValue.PINNED))
        result = store.query_by_family_key("ds1", "fk1", {"fk1": "a1"})
        assert len(result) == 1
        assert result[0].status == AssertionStatusValue.PINNED

    def test_separate_datasources(self, store: StatusEventStore) -> None:
        store.append("ds1", _event("a1", AssertionStatusValue.PINNED))
        store.append("ds2", _event("a1", AssertionStatusValue.REJECTED))
        ds1 = store.query_latest_by_assertion_id("ds1", "a1")
        ds2 = store.query_latest_by_assertion_id("ds2", "a1")
        assert ds1 is not None and ds1.status == AssertionStatusValue.PINNED
        assert ds2 is not None and ds2.status == AssertionStatusValue.REJECTED


class TestEffectiveStatus:
    def test_default_auto_when_no_events(
        self, store: StatusEventStore
    ) -> None:
        status = effective_status(store, "ds1", "a1")
        assert status == AssertionStatusValue.AUTO

    def test_returns_latest_event_status(
        self, store: StatusEventStore
    ) -> None:
        store.append("ds1", _event("a1", AssertionStatusValue.ACCEPTED))
        store.append("ds1", _event("a1", AssertionStatusValue.PINNED))
        status = effective_status(store, "ds1", "a1")
        assert status == AssertionStatusValue.PINNED

    def test_rejected_is_effective(
        self, store: StatusEventStore
    ) -> None:
        store.append("ds1", _event("a1", AssertionStatusValue.REJECTED))
        status = effective_status(store, "ds1", "a1")
        assert status == AssertionStatusValue.REJECTED
