import pytest
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit

from neo4j.exceptions import ClientError, TransientError

from sema.graph.loader import GraphLoader
from sema.graph.retry_utils import (
    BASE_RETRY_DELAY_SECONDS,
    MAX_RETRY_ATTEMPTS,
    run_with_retry,
    statement_summary,
)


class _FakeOp:
    def __init__(self, errors, result="ok"):
        self._errors = list(errors)
        self._result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return self._result


def _no_sleep():
    delays = []
    return delays.append, delays


def test_succeeds_after_two_transient_errors():
    op = _FakeOp([TransientError("deadlock"), TransientError("deadlock")])
    sleeper, delays = _no_sleep()
    result = run_with_retry(op, summary="MERGE", sleep=sleeper)
    assert result == "ok"
    assert op.calls == 3
    assert len(delays) == 2


def test_persistent_transient_raises_after_max_attempts():
    op = _FakeOp([TransientError("deadlock")] * 10)
    sleeper, _ = _no_sleep()
    with pytest.raises(TransientError):
        run_with_retry(op, summary="MERGE", sleep=sleeper)
    assert op.calls == MAX_RETRY_ATTEMPTS


def test_client_error_not_retried():
    op = _FakeOp([ClientError("bad syntax")])
    sleeper, delays = _no_sleep()
    with pytest.raises(ClientError):
        run_with_retry(op, summary="MERGE", sleep=sleeper)
    assert op.calls == 1
    assert delays == []


def test_exponential_backoff_delays():
    op = _FakeOp([TransientError("x"), TransientError("x")])
    sleeper, delays = _no_sleep()
    run_with_retry(op, summary="MERGE", sleep=sleeper)
    assert delays == [
        BASE_RETRY_DELAY_SECONDS,
        BASE_RETRY_DELAY_SECONDS * 2,
    ]


def test_statement_summary_truncates_to_first_line():
    summary = statement_summary("MERGE (n:Foo)\nSET n.x = 1", limit=20)
    assert summary == "MERGE (n:Foo)"


class TestLoaderRetries:
    def _loader_raising(self, errors):
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(
            return_value=session
        )
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        session.run.side_effect = errors
        return GraphLoader(driver), session

    def test_run_retries_transient_twice_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("sema.graph.retry_utils._sleep", lambda _s: None)
        loader, session = self._loader_raising(
            [TransientError("deadlock"), TransientError("deadlock"), None]
        )
        loader._run("MERGE (n:Foo)")
        assert session.run.call_count == 3

    def test_run_persistent_transient_raises(self, monkeypatch):
        monkeypatch.setattr("sema.graph.retry_utils._sleep", lambda _s: None)
        loader, session = self._loader_raising(
            [TransientError("deadlock")] * 10
        )
        with pytest.raises(TransientError):
            loader._run("MERGE (n:Foo)")
        assert session.run.call_count == MAX_RETRY_ATTEMPTS

    def test_run_client_error_propagates_immediately(self, monkeypatch):
        monkeypatch.setattr("sema.graph.retry_utils._sleep", lambda _s: None)
        loader, session = self._loader_raising([ClientError("bad")])
        with pytest.raises(ClientError):
            loader._run("MERGE (n:Foo)")
        assert session.run.call_count == 1

    def test_run_read_retries_then_returns_rows(self, monkeypatch):
        monkeypatch.setattr("sema.graph.retry_utils._sleep", lambda _s: None)
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(
            return_value=session
        )
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        session.run.side_effect = [TransientError("deadlock"), [{"n": 1}]]
        loader = GraphLoader(driver)
        rows = loader._run_read("MATCH (n) RETURN n")
        assert rows == [{"n": 1}]
        assert session.run.call_count == 2
