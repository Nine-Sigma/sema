"""The run registry: one SQLite row per ingestion run, beside the ledger directories.

SQLite, not DuckDB staging, because staging is per source while the registry spans
sources and must stay readable while a run holds a staging file (design Decision 12).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_STATUSES = frozenset(
    {"running", "waiting_review", "approved", "rejected", "failed", "complete"}
)
RUN_REASONS = frozenset({"budget_exceeded", "escalate"})
INITIAL_STATUS = "running"
FILTER_COLUMNS = ("status", "stage", "source_identity")
RECORD_COLUMNS = (
    "run_id",
    "source_identity",
    "stage",
    "status",
    "approver",
    "started_at",
    "updated_at",
    "model_calls",
    "tokens",
    "cost",
    "reason",
)
CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    source_identity TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    approver TEXT,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    model_calls INTEGER NOT NULL DEFAULT 0,
    tokens INTEGER NOT NULL DEFAULT 0,
    cost REAL NOT NULL DEFAULT 0,
    reason TEXT
)
"""
ADD_REASON_COLUMN = "ALTER TABLE runs ADD COLUMN reason TEXT"

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    source: str
    stage: str
    status: str
    approver: str | None
    started_at: datetime
    updated_at: datetime
    model_calls: int
    tokens: int
    cost: float
    reason: str | None = None


def _record(row: tuple[Any, ...]) -> RunRecord:
    run_id, source, stage, status, approver, started, updated, calls, tokens, cost, reason = row
    return RunRecord(
        run_id=run_id,
        source=source,
        stage=stage,
        status=status,
        approver=approver,
        started_at=datetime.fromisoformat(started),
        updated_at=datetime.fromisoformat(updated),
        model_calls=calls,
        tokens=tokens,
        cost=cost,
        reason=reason,
    )


class Registry:
    def __init__(self, path: Path, clock: Clock = _utc_now) -> None:
        self.path = path
        self._clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(CREATE_TABLE)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
            if "reason" not in columns:
                conn.execute(ADD_REASON_COLUMN)

    def register(self, run_id: str, *, source: str, stage: str) -> None:
        now = self._now()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO runs"
                    " (run_id, source_identity, stage, status, started_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, source, stage, INITIAL_STATUS, now, now),
                )
        except sqlite3.IntegrityError as err:
            raise ValueError(f"run id {run_id} is already registered") from err

    def advance(
        self, run_id: str, stage: str, status: str, reason: str | None = None
    ) -> None:
        """Move a run on; a call without a reason clears any reason stored before."""
        if status not in RUN_STATUSES:
            expected = ", ".join(sorted(RUN_STATUSES))
            raise ValueError(f"unknown run status {status!r}; expected one of {expected}")
        if reason is not None and reason not in RUN_REASONS:
            expected = ", ".join(sorted(RUN_REASONS))
            raise ValueError(f"unknown run reason {reason!r}; expected one of {expected}")
        self._update(
            run_id,
            "stage = ?, status = ?, reason = ?, updated_at = ?",
            (stage, status, reason, self._now()),
        )

    def count_call(self, run_id: str, tokens: int, cost: float) -> None:
        self._update(
            run_id,
            "model_calls = model_calls + 1, tokens = tokens + ?, cost = cost + ?,"
            " updated_at = ?",
            (tokens, cost, self._now()),
        )

    def get(self, run_id: str) -> RunRecord:
        query = f"SELECT {', '.join(RECORD_COLUMNS)} FROM runs WHERE run_id = ?"
        with self._connect() as conn:
            row = conn.execute(query, (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"run id {run_id} is not registered")
        return _record(row)

    def list(
        self, status: str | None = None, stage: str | None = None, source: str | None = None
    ) -> tuple[RunRecord, ...]:
        wanted = dict(zip(FILTER_COLUMNS, (status, stage, source), strict=True))
        filters = {column: value for column, value in wanted.items() if value is not None}
        where = " AND ".join(f"{column} = ?" for column in filters)
        query = f"SELECT {', '.join(RECORD_COLUMNS)} FROM runs"
        if where:
            query += f" WHERE {where}"
        query += " ORDER BY started_at DESC, rowid DESC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(filters.values())).fetchall()
        return tuple(_record(row) for row in rows)

    def _update(self, run_id: str, assignments: str, params: tuple[object, ...]) -> None:
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE runs SET {assignments} WHERE run_id = ?", (*params, run_id)
            )
        if cursor.rowcount == 0:
            raise KeyError(f"run id {run_id} is not registered")

    def _now(self) -> str:
        return self._clock().isoformat()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path)) as conn, conn:
            yield conn
