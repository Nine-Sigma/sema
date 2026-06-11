"""Retry wrapper for transient Neo4j errors.

MERGE on nodes shared across concurrently-materialized tables can hit
``neo4j.exceptions.TransientError`` (lock acquisition / deadlock). These are
safe to retry; ``ClientError`` (syntax, constraint) is not and propagates
immediately. Backoff is exponential off ``BASE_RETRY_DELAY_SECONDS``.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from neo4j.exceptions import TransientError

from sema.log import logger

MAX_RETRY_ATTEMPTS = 3
BASE_RETRY_DELAY_SECONDS = 0.2

T = TypeVar("T")


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def statement_summary(cypher: str, limit: int = 80) -> str:
    stripped = cypher.strip()
    first_line = stripped.splitlines()[0] if stripped else ""
    return first_line[:limit]


def run_with_retry(
    operation: Callable[[], T],
    summary: str,
    sleep: Callable[[float], None] | None = None,
) -> T:
    do_sleep = sleep if sleep is not None else _sleep
    attempt = 1
    while True:
        try:
            return operation()
        except TransientError as exc:
            if attempt >= MAX_RETRY_ATTEMPTS:
                logger.warning(
                    "Neo4j transient error on '{}' exhausted {} attempts: {}",
                    summary, MAX_RETRY_ATTEMPTS, exc,
                )
                raise
            delay = BASE_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Neo4j transient error on '{}' (attempt {}/{}); "
                "retrying in {}s",
                summary, attempt, MAX_RETRY_ATTEMPTS, delay,
            )
            do_sleep(delay)
            attempt += 1
