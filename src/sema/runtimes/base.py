"""SQL runtime protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SQLRuntime(Protocol):
    """Protocol for SQL execution backends."""

    @property
    def dialect(self) -> str: ...

    def execute(self, sql: str) -> dict[str, Any]: ...

    def explain(self, sql: str) -> str: ...

    def close(self) -> None: ...
