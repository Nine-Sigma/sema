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
    cost REAL NOT NULL DEFAULT 0
)
"""

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


def _record(row: tuple[Any, ...]) -> RunRecord:
    run_id, source, stage, status, approver, started, updated, calls, tokens, cost = row
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
    )


class Registry:
    def __init__(self, path: Path, clock: Clock = _utc_now) -> None:
        self.path = path
        self._clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(CREATE_TABLE)

    def register(self, run_id: str, *, source: str, stage: str) -> None:
        now = self._now()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO runs"
                    " (run_id, source_identity, stage, status, started_at, updated_at)"
                    " SELECT ?, ?, ?, ?, ?, ?",
                    (run_id, source, stage, INITIAL_STATUS, now, now),
                )
        except sqlite3.IntegrityError as err:
            raise ValueError(f"run id {run_id} is already registered") from err

    def advance(self, run_id: str, stage: str, status: str) -> None:
        if status not in RUN_STATUSES:
            expected = ", ".join(sorted(RUN_STATUSES))
            raise ValueError(f"unknown run status {status!r}; expected one of {expected}")
        self._update(
            run_id,
            "stage = ?, status = ?, updated_at = ?",
            (stage, status, self._now()),
        )

    def count_call(self, run_id: str, tokens: int, cost: float) -> None:
        self._update(
            run_id,
            "model_calls = model_calls + 1, tokens = tokens + ?, cost = cost + ?,"
            " updated_at = ?",
            (tokens, cost, self._now()),
        )

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
