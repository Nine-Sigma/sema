"""Unit tests for consumer protocol, registry, and base types."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


class TestResolveConsumer:
    def test_resolve_known_consumer(self):
        from sema.consumers import resolve_consumer

        consumer = resolve_consumer("nl2sql")
        assert consumer.name == "nl2sql"
        assert "plan" in consumer.capabilities

    def test_resolve_unknown_raises(self):
        from sema.consumers import resolve_consumer

        with pytest.raises(ValueError, match="unknown"):
            resolve_consumer("unknown")

    def test_error_lists_available(self):
        from sema.consumers import resolve_consumer

        with pytest.raises(ValueError, match="nl2sql"):
            resolve_consumer("nonexistent")


class TestConsumerResult:
    def test_construction(self):
        from sema.consumers.base import ConsumerResult

        result = ConsumerResult(
            artifact="SELECT 1",
            valid=True,
            errors=[],
            attempts=1,
            data={},
            summary=None,
        )
        assert result.artifact == "SELECT 1"
        assert result.valid is True
        assert result.attempts == 1

    def test_failed_result(self):
        from sema.consumers.base import ConsumerResult

        result = ConsumerResult(
            artifact="BAD SQL",
            valid=False,
            errors=["syntax error"],
            attempts=3,
            data={},
            summary=None,
        )
        assert result.valid is False
        assert len(result.errors) == 1


class TestConsumerDeps:
    def test_deps_with_all(self):
        from sema.consumers.base import ConsumerDeps

        llm = MagicMock()
        runtime = MagicMock()
        deps = ConsumerDeps(llm=llm, sql_runtime=runtime)
        assert deps.llm is llm
        assert deps.sql_runtime is runtime

    def test_deps_defaults_to_none(self):
        from sema.consumers.base import ConsumerDeps

        deps = ConsumerDeps()
        assert deps.llm is None
        assert deps.sql_runtime is None

    def test_deps_frozen(self):
        from sema.consumers.base import ConsumerDeps

        deps = ConsumerDeps()
        with pytest.raises(AttributeError):
            deps.llm = MagicMock()  # type: ignore[misc]


class TestConsumerRequest:
    def test_construction(self):
        from sema.consumers.base import ConsumerRequest

        req = ConsumerRequest(question="show patients", operation="plan")
        assert req.question == "show patients"
        assert req.operation == "plan"


class TestContextProfile:
    def test_cache_key_stub(self):
        from sema.consumers.base import ContextProfile

        profile = ContextProfile()
        assert profile.cache_key() == ""
