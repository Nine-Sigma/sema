from __future__ import annotations

import threading
import time
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open and calls are rejected."""


class CircuitBreaker:
    """Thread-safe circuit breaker for LLM invocations.

    Tracks logical invocation failures (not internal retries).
    Opens after failure_threshold consecutive failures, transitions
    to half-open after recovery_timeout seconds, and closes after
    success_threshold consecutive successes in half-open state.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold

        self._lock = threading.Lock()
        self._failure_count = 0
        self._success_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float = 0.0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    def check(self) -> None:
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return
            if self._state == CircuitState.HALF_OPEN:
                return
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                return
            raise CircuitOpenError(
                f"Circuit breaker is open ({self._failure_count} failures, "
                f"{self._recovery_timeout - elapsed:.1f}s until probe)"
            )

    def record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
            elif self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
