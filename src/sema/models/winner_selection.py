"""Winner selection: status-tiered with scoring within AUTO tier only.

Algorithm:
1. Exclude REJECTED and SUPERSEDED
2. If any PINNED -> pick most recent PINNED
3. Else if any ACCEPTED -> pick most recent ACCEPTED
4. Else among AUTO -> score by (normalized_precedence * 0.4) + (confidence * 0.6)
"""

from __future__ import annotations

from sema.models.assertions import Assertion
from sema.models.constants import source_precedence
from sema.models.lifecycle import AssertionStatusValue, StatusEvent
from sema.models.status_store import StatusEventStore, effective_status


def _auto_score(assertion: Assertion) -> float:
    """Compute AUTO-tier score: weighted precedence + confidence."""
    normalized_prec = source_precedence(assertion.source) / 100.0
    return (normalized_prec * 0.4) + (assertion.confidence * 0.6)


def select_winner(
    family_assertions: list[Assertion],
    store: StatusEventStore,
    datasource_id: str,
) -> Assertion | None:
    """Select the winning assertion from an assertion family.

    Status priority is absolute: PINNED > ACCEPTED > AUTO scoring.
    The scoring formula ONLY applies within the AUTO tier.
    """
    if not family_assertions:
        return None

    # Compute effective status for each assertion
    with_status: list[tuple[Assertion, AssertionStatusValue]] = []
    for a in family_assertions:
        status = effective_status(store, datasource_id, a.id)
        with_status.append((a, status))

    # Exclude REJECTED and SUPERSEDED
    active = [
        (a, s) for a, s in with_status
        if s not in (
            AssertionStatusValue.REJECTED,
            AssertionStatusValue.SUPERSEDED,
        )
    ]
    if not active:
        return None

    # PINNED wins — pick most recent
    pinned = [
        (a, s) for a, s in active
        if s == AssertionStatusValue.PINNED
    ]
    if pinned:
        return max(pinned, key=lambda t: t[0].observed_at)[0]

    # ACCEPTED wins — pick most recent
    accepted = [
        (a, s) for a, s in active
        if s == AssertionStatusValue.ACCEPTED
    ]
    if accepted:
        return max(accepted, key=lambda t: t[0].observed_at)[0]

    # AUTO tier — score and pick highest
    auto = [a for a, s in active if s == AssertionStatusValue.AUTO]
    if not auto:
        return None
    return max(auto, key=_auto_score)
