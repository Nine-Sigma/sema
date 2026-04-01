from __future__ import annotations

import re

from sema.models.context import SemanticContextObject


class ContextStore:
    """In-process SCO cache keyed by (session_id, normalized_query, consumer, graph_version_hash)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, str, str], SemanticContextObject] = {}

    @staticmethod
    def _normalize_query(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().lower())

    def _key(
        self, session_id: str, query: str, consumer: str, graph_version: str
    ) -> tuple[str, str, str, str]:
        return (session_id, self._normalize_query(query), consumer, graph_version)

    def get(
        self, session_id: str, query: str, consumer: str, graph_version: str
    ) -> SemanticContextObject | None:
        return self._cache.get(
            self._key(session_id, query, consumer, graph_version)
        )

    def put(
        self,
        session_id: str,
        query: str,
        consumer: str,
        graph_version: str,
        sco: SemanticContextObject,
    ) -> None:
        self._cache[self._key(session_id, query, consumer, graph_version)] = sco

    def invalidate(self) -> None:
        self._cache.clear()
