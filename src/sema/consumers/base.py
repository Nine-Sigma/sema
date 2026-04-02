"""Consumer protocol and base types for the consumer framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sema.models.context import SemanticContextObject


@dataclass(frozen=True)
class ContextProfile:
    """Consumer context profile — stub for phase 1.

    Future phases will use this to tell the engine what
    SCO shape the consumer needs.
    """

    def cache_key(self) -> str:
        return ""


@dataclass(frozen=True)
class ConsumerRequest:
    """Request passed from the orchestrator to a consumer."""

    question: str
    operation: str


@dataclass(frozen=True)
class ConsumerDeps:
    """Narrow dependency injection for consumers."""

    llm: Any | None = None
    sql_runtime: Any | None = None


@dataclass
class ConsumerResult:
    """Generic result returned by Consumer.run()."""

    artifact: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    attempts: int = 1
    data: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None


@runtime_checkable
class Consumer(Protocol):
    """Thin uniform interface for orchestrator dispatch."""

    name: str
    capabilities: set[str]

    def context_profile(self) -> ContextProfile: ...

    def run(
        self,
        request: ConsumerRequest,
        sco: SemanticContextObject,
        deps: ConsumerDeps,
    ) -> ConsumerResult: ...
