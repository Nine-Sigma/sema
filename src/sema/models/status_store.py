"""StatusEvent store: append-only persistence for assertion status transitions.

JSON-file backend per datasource_id. Designed for easy replacement
with a database backend later.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sema.models.lifecycle import AssertionStatusValue, StatusEvent


class StatusEventStore:
    """Append-only store for StatusEvents, keyed by datasource_id."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _store_path(self, datasource_id: str) -> Path:
        safe_name = datasource_id.replace("/", "_").replace(":", "_")
        return self._base_dir / f"status_events_{safe_name}.jsonl"

    def append(self, datasource_id: str, event: StatusEvent) -> None:
        """Append a status event to the store."""
        path = self._store_path(datasource_id)
        with path.open("a") as f:
            f.write(event.model_dump_json() + "\n")

    def query_by_assertion_id(
        self, datasource_id: str, assertion_id: str
    ) -> list[StatusEvent]:
        """Return all events for a given assertion_id, oldest first."""
        events = self._load_all(datasource_id)
        return [e for e in events if e.assertion_id == assertion_id]

    def query_latest_by_assertion_id(
        self, datasource_id: str, assertion_id: str
    ) -> StatusEvent | None:
        """Return the most recent event for an assertion_id, or None."""
        matching = self.query_by_assertion_id(datasource_id, assertion_id)
        return matching[-1] if matching else None

    def query_by_family_key(
        self,
        datasource_id: str,
        family_key: str,
        family_index: dict[str, str] | None = None,
    ) -> list[StatusEvent]:
        """Return events matching a family key via the family index.

        ``family_index`` maps family_key -> assertion_id for the
        current run's assertions. This enables cross-run replay.
        """
        if not family_index:
            return []
        assertion_id = family_index.get(family_key)
        if not assertion_id:
            return []
        return self.query_by_assertion_id(datasource_id, assertion_id)

    def bulk_load(
        self, datasource_id: str, events: list[StatusEvent]
    ) -> None:
        """Append multiple events at once."""
        if not events:
            return
        path = self._store_path(datasource_id)
        with path.open("a") as f:
            for event in events:
                f.write(event.model_dump_json() + "\n")

    def _load_all(self, datasource_id: str) -> list[StatusEvent]:
        path = self._store_path(datasource_id)
        if not path.exists():
            return []
        events: list[StatusEvent] = []
        for line in path.read_text().strip().split("\n"):
            if line:
                events.append(StatusEvent.model_validate_json(line))
        return events


def effective_status(
    store: StatusEventStore,
    datasource_id: str,
    assertion_id: str,
) -> AssertionStatusValue:
    """Return the effective status for an assertion.

    Reads the most recent StatusEvent. Returns AUTO if none exists.
    """
    event = store.query_latest_by_assertion_id(
        datasource_id, assertion_id
    )
    if event is None:
        return AssertionStatusValue.AUTO
    return event.status
