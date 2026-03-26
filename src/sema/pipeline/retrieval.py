from __future__ import annotations

import logging
from typing import Any

from sema.graph.queries import CypherQueries
from sema.models.constants import MATCH_TYPE_BOOST
from sema.models.context import SemanticCandidateSet
from sema.pipeline.retrieval_utils import (
    _expand_ancestry,
    _expand_joins,
    _expand_metrics,
    _expand_physical,
    _expand_values,
    merge_and_rank_candidates,
)

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Hybrid retrieval: vector + lexical + graph traversal."""

    def __init__(self, driver: Any, embedder: Any = None) -> None:
        self._driver = driver
        self._embedder = embedder

    def _run_query(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            results = session.run(query, **params)
            return [r.data() for r in results]

    def vector_search(
        self, question: str, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Search across all embedded node types."""
        if not self._embedder:
            return []

        if hasattr(self._embedder, "embed_documents"):
            embedding = self._embedder.embed_documents([question])[0]
        else:
            embedding = self._embedder.encode([question])[0]
        candidates = []

        for index in [
            "entity_embedding_index", "property_embedding_index",
            "term_embedding_index", "alias_embedding_index",
            "metric_embedding_index",
        ]:
            try:
                query = CypherQueries.vector_search(index, top_k)
                results = self._run_query(
                    query, embedding=list(embedding),
                )
                for r in results:
                    node = r.get("node", {})
                    candidates.append({
                        **node,
                        "score": r.get("score", 0.0),
                        "match_type": "vector",
                        "index": index,
                    })
            except Exception as e:
                logger.debug(f"Vector search on {index} failed: {e}")

        return candidates

    def expand_from_entities(
        self, entity_names: list[str]
    ) -> dict[str, Any]:
        """Expand matched entities via graph traversal."""
        physical = _expand_physical(self, entity_names)

        table_names = [
            r["table_name"] for r in physical if r.get("table_name")
        ]

        joins = _expand_joins(self, table_names)
        values = _expand_values(self, physical)
        metrics = _expand_metrics(self, entity_names)
        ancestry = _expand_ancestry(self, entity_names)

        return {
            "physical": physical,
            "joins": joins,
            "values": values,
            "ancestry": ancestry,
            "metrics": metrics,
        }

    def retrieve(
        self, question: str, top_k: int = 10
    ) -> SemanticCandidateSet:
        """Full hybrid retrieval pipeline."""
        vector_candidates = self.vector_search(question, top_k)
        ranked = merge_and_rank_candidates(vector_candidates)

        entity_names = [
            c["name"] for c in ranked
            if c.get("name") and "entity" in c.get("index", "")
        ]

        expansion = self.expand_from_entities(entity_names)
        all_candidates = []

        for r in expansion.get("physical", []):
            all_candidates.append({
                "type": "entity",
                "name": next(
                    (n for n in entity_names), r["table_name"]
                ),
                "table": r["table_name"],
                "schema": r.get("schema_name", ""),
                "catalog": r.get("catalog", ""),
                "description": None,
                "confidence": 0.8,
                "source": "retrieval",
                "columns": r.get("columns", []),
            })

        for j in expansion.get("joins", []):
            all_candidates.append({"type": "join", **j})

        for v in expansion.get("values", []):
            all_candidates.append({
                "type": "value",
                "property_name": v.get("property", ""),
                "column": v.get("column", ""),
                "table": v.get("table", ""),
                "code": v.get("code", ""),
                "label": v.get("label", ""),
            })

        for m in expansion.get("metrics", []):
            all_candidates.append({"type": "metric", **m})

        return SemanticCandidateSet(
            query=question,
            candidates=all_candidates,
        )
