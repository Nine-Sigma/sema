from __future__ import annotations

import threading
import time

import pytest

pytestmark = pytest.mark.unit

from sema.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitOpensAfterThreshold:
    def test_five_failures_opens_circuit(self) -> None:
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestSingleInvokeCountsAsOne:
    def test_one_invoke_with_retries_is_one_failure(self) -> None:
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 1


class TestCircuitHalfOpensAfterTimeout:
    def test_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        cb.check()
        assert cb.state == CircuitState.HALF_OPEN


class TestCircuitClosesOnRecovery:
    def test_two_successes_in_half_open_closes(self) -> None:
        cb = CircuitBreaker(
            failure_threshold=2, recovery_timeout=0.1, success_threshold=2,
        )
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        cb.check()
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerThreadSafe:
    def test_concurrent_record_failure(self) -> None:
        cb = CircuitBreaker(failure_threshold=100, recovery_timeout=60)
        barrier = threading.Barrier(10)

        def worker() -> None:
            barrier.wait()
            for _ in range(10):
                cb.record_failure()

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cb._failure_count == 100
        assert cb.state == CircuitState.OPEN


class TestCircuitOpenRaisesImmediately:
    def test_check_raises_when_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            cb.check()
