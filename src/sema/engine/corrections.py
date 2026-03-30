"""Human feedback loop: correction-to-StatusEvent application logic.

Corrections are append-only. Machine assertions are never mutated.
Human corrections create StatusEvents or new assertion records.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.family_key import family_key
from sema.models.lifecycle import AssertionStatusValue, StatusEvent
from sema.models.status_store import StatusEventStore


def confirm_assertion(
    store: StatusEventStore,
    datasource_id: str,
    assertion_id: str,
) -> StatusEvent:
    """Human confirms a machine assertion -> PINNED."""
    event = StatusEvent(
        assertion_id=assertion_id,
        status=AssertionStatusValue.PINNED,
        actor="human",
        timestamp=datetime.now(timezone.utc),
    )
    store.append(datasource_id, event)
    return event


def reject_assertion(
    store: StatusEventStore,
    datasource_id: str,
    assertion_id: str,
    reason: str | None = None,
) -> StatusEvent:
    """Human rejects a machine assertion -> REJECTED."""
    event = StatusEvent(
        assertion_id=assertion_id,
        status=AssertionStatusValue.REJECTED,
        actor="human",
        reason=reason,
        timestamp=datetime.now(timezone.utc),
    )
    store.append(datasource_id, event)
    return event


def relabel_assertion(
    store: StatusEventStore,
    datasource_id: str,
    old_assertion_id: str,
    new_subject_ref: str,
    new_predicate: AssertionPredicate,
    new_payload: dict[str, Any],
    run_id: str,
    reason: str | None = None,
) -> tuple[StatusEvent, Assertion, StatusEvent]:
    """Human relabels: reject old + create new human assertion + pin it.

    Returns (reject_event, new_assertion, pin_event).
    """
    # Reject old
    reject_event = reject_assertion(
        store, datasource_id, old_assertion_id, reason=reason,
    )

    # Create new human assertion
    new_assertion = Assertion(
        id=str(uuid.uuid4()),
        subject_ref=new_subject_ref,
        predicate=new_predicate,
        payload=new_payload,
        source="human",
        confidence=1.0,
        run_id=run_id,
        observed_at=datetime.now(timezone.utc),
    )

    # Pin the new assertion
    pin_event = StatusEvent(
        assertion_id=new_assertion.id,
        status=AssertionStatusValue.PINNED,
        actor="human",
        reason=reason,
        timestamp=datetime.now(timezone.utc),
    )
    store.append(datasource_id, pin_event)

    return reject_event, new_assertion, pin_event


def replay_corrections(
    store: StatusEventStore,
    datasource_id: str,
    assertions: list[Assertion],
    stored_corrections: dict[str, AssertionStatusValue],
) -> list[StatusEvent]:
    """Replay stored corrections against new run's assertions.

    ``stored_corrections`` maps family_key -> status to apply.
    Matches new assertions by family key and emits StatusEvents.
    """
    emitted: list[StatusEvent] = []
    for a in assertions:
        fk = family_key(
            a.subject_ref, a.predicate, a.payload, a.object_ref,
        )
        target_status = stored_corrections.get(fk)
        if target_status is None:
            continue
        event = StatusEvent(
            assertion_id=a.id,
            status=target_status,
            actor="human_replay",
            reason="Replayed from stored corrections",
            timestamp=datetime.now(timezone.utc),
        )
        store.append(datasource_id, event)
        emitted.append(event)
    return emitted
