"""Consumer registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sema.consumers.base import Consumer


def _build_registry() -> dict[str, type[Consumer]]:
    from sema.consumers.nl2sql.consumer import NL2SQLConsumer

    return {
        "nl2sql": NL2SQLConsumer,
    }


def resolve_consumer(name: str) -> Consumer:
    """Resolve a consumer by name. Raises ValueError for unknown names."""
    registry = _build_registry()
    cls = registry.get(name)
    if cls is None:
        available = sorted(registry.keys())
        raise ValueError(
            f"Unknown consumer: {name!r}. "
            f"Available: {available}"
        )
    return cls()
