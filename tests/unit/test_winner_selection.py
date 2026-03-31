"""Tests for status-tiered winner selection."""

import pytest
from datetime import datetime, timezone
from pathlib import Path

from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.lifecycle import AssertionStatusValue, StatusEvent
from sema.models.status_store import StatusEventStore
from sema.models.winner_selection import select_winner, _auto_score

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path: Path) -> StatusEventStore:
    return StatusEventStore(tmp_path / "status_events")


def _assertion(
    id: str,
    source: str = "llm_interpretation",
    confidence: float = 0.8,
) -> Assertion:
    return Assertion(
        id=id,
        subject_ref="col_X",
        predicate=AssertionPredicate.VOCABULARY_MATCH,
        payload={"value": "ICD-10"},
        source=source,
        confidence=confidence,
        run_id="run-1",
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _pin(store: StatusEventStore, assertion_id: str) -> None:
    store.append("ds1", StatusEvent(
        assertion_id=assertion_id,
        status=AssertionStatusValue.PINNED,
        actor="human",
        timestamp=datetime.now(timezone.utc),
    ))


def _reject(store: StatusEventStore, assertion_id: str) -> None:
    store.append("ds1", StatusEvent(
        assertion_id=assertion_id,
        status=AssertionStatusValue.REJECTED,
        actor="human",
        timestamp=datetime.now(timezone.utc),
    ))


def _accept(store: StatusEventStore, assertion_id: str) -> None:
    store.append("ds1", StatusEvent(
        assertion_id=assertion_id,
        status=AssertionStatusValue.ACCEPTED,
        actor="human",
        timestamp=datetime.now(timezone.utc),
    ))


def _supersede(store: StatusEventStore, assertion_id: str) -> None:
    store.append("ds1", StatusEvent(
        assertion_id=assertion_id,
        status=AssertionStatusValue.SUPERSEDED,
        actor="machine",
        timestamp=datetime.now(timezone.utc),
    ))


class TestStatusPriority:
    def test_pinned_beats_high_confidence_auto(
        self, store: StatusEventStore
    ) -> None:
        pinned = _assertion("a1", confidence=0.5)
        auto = _assertion("a2", source="atlan", confidence=0.99)
        _pin(store, "a1")
        winner = select_winner([pinned, auto], store, "ds1")
        assert winner is not None
        assert winner.id == "a1"

    def test_accepted_beats_all_auto(
        self, store: StatusEventStore
    ) -> None:
        accepted = _assertion("a1", confidence=0.5)
        auto_high = _assertion("a2", source="atlan", confidence=0.99)
        _accept(store, "a1")
        winner = select_winner([accepted, auto_high], store, "ds1")
        assert winner is not None
        assert winner.id == "a1"

    def test_rejected_excluded_entirely(
        self, store: StatusEventStore
    ) -> None:
        rejected = _assertion("a1", confidence=0.99)
        auto = _assertion("a2", confidence=0.5)
        _reject(store, "a1")
        winner = select_winner([rejected, auto], store, "ds1")
        assert winner is not None
        assert winner.id == "a2"

    def test_superseded_excluded_entirely(
        self, store: StatusEventStore
    ) -> None:
        superseded = _assertion("a1", confidence=0.99)
        auto = _assertion("a2", confidence=0.5)
        _supersede(store, "a1")
        winner = select_winner([superseded, auto], store, "ds1")
        assert winner is not None
        assert winner.id == "a2"

    def test_all_rejected_returns_none(
        self, store: StatusEventStore
    ) -> None:
        a1 = _assertion("a1")
        a2 = _assertion("a2")
        _reject(store, "a1")
        _reject(store, "a2")
        winner = select_winner([a1, a2], store, "ds1")
        assert winner is None

    def test_empty_family_returns_none(
        self, store: StatusEventStore
    ) -> None:
        winner = select_winner([], store, "ds1")
        assert winner is None


class TestAutoTierScoring:
    def test_uc_beats_llm_close_confidence(
        self, store: StatusEventStore
    ) -> None:
        """UC prec=40 conf=0.8 (score 0.64) > LLM prec=20 conf=0.9 (score 0.62)"""
        uc = _assertion("a1", source="unity_catalog", confidence=0.8)
        llm = _assertion("a2", source="llm_interpretation", confidence=0.9)
        winner = select_winner([uc, llm], store, "ds1")
        assert winner is not None
        assert winner.id == "a1"

    def test_llm_beats_uc_large_confidence_gap(
        self, store: StatusEventStore
    ) -> None:
        """LLM prec=20 conf=0.95 (score 0.65) > UC prec=40 conf=0.5 (score 0.46)"""
        uc = _assertion("a1", source="unity_catalog", confidence=0.5)
        llm = _assertion("a2", source="llm_interpretation", confidence=0.95)
        winner = select_winner([uc, llm], store, "ds1")
        assert winner is not None
        assert winner.id == "a2"

    def test_score_formula_values(self) -> None:
        uc = _assertion("a1", source="unity_catalog", confidence=0.8)
        llm = _assertion("a2", source="llm_interpretation", confidence=0.9)
        # UC: (40/100 * 0.4) + (0.8 * 0.6) = 0.16 + 0.48 = 0.64
        assert abs(_auto_score(uc) - 0.64) < 0.001
        # LLM: (20/100 * 0.4) + (0.9 * 0.6) = 0.08 + 0.54 = 0.62
        assert abs(_auto_score(llm) - 0.62) < 0.001
