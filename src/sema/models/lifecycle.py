"""Assertion lifecycle: StatusEvent model and effective status computation.

Assertion records are immutable. Status changes are tracked via an
append-only StatusEvent log. The effective status of an assertion is
the most recent StatusEvent for that assertion ID (default AUTO).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


class AssertionStatusValue(str, Enum):
    """Effective status values for assertions."""

    AUTO = "auto"
    ACCEPTED = "accepted"
    PINNED = "pinned"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class StatusEvent(BaseModel):
    """An immutable record of a status transition for one assertion."""

    assertion_id: str
    status: AssertionStatusValue
    actor: str  # "machine" or "human"
    reason: str | None = None
    timestamp: datetime
