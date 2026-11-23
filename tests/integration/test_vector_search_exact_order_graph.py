"""Contract F2, vector boundary (N4): a vector read returns the exact top_k under a total order.

The neutral fixture is built with the embedding stage on through a stand-in embedder that
derives each vector from a SHA-256 of its text. For every probe and every vector index,
`RetrievalEngine.vector_search` must return the top_k nodes of exact cosine similarity,
computed here from the stored vectors, ordered by score and then by the node's build-stable
identity. An approximate or quantized index read alone does not meet this.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from sema.models.config import Neo4jConfig
from sema.pipeline.orchestrate import run_build
from sema.pipeline.retrieval import RetrievalEngine
from tests.support.connector_fixture import (
    MAIN_TABLES,
    fake_model_environment,
    fixture_build_config,
    write_fixture_database,
)
from tests.support.connector_fixture_context import ASKED_HIERARCHY_QUESTION

pytestmark = pytest.mark.integration

IDENTITY = ("ref", "schema_name", "entity_name", "name", "vocabulary_name", "code",
            "target_key", "text")


class HashEmbedder:
    """One text gives one vector on every build; two texts give two vectors."""

    dimensions = 64

    def vector(self, text: str) -> list[float]:
        raw = b""
        while len(raw) < self.dimensions:
            raw += hashlib.sha256(f"{len(raw)}:{text}".encode()).digest()
        values = [byte / 127.5 - 1.0 for byte in raw[: self.dimensions]]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vector(t) for t in texts]


def _neo4j_config() -> Neo4jConfig:
    return Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=SecretStr(os.getenv("NEO4J_PASSWORD", "graphrag")),
    )


def _drop_vector_indexes(driver: Any) -> None:
    with driver.session() as session:
        for row in session.run("SHOW VECTOR INDEXES YIELD name RETURN name").data():
            session.run(f"DROP INDEX `{row['name']}` IF EXISTS").consume()


def _build(driver: Any, tmp_path: Path) -> dict[str, str]:
    _drop_vector_indexes(driver)
    config = fixture_build_config(_neo4j_config(), database=write_fixture_database(tmp_path))
    config = config.model_copy(update={"skip_embeddings": False})
    with fake_model_environment(), patch(
        "sema.pipeline.orchestrate_utils._get_embedder", lambda _config: HashEmbedder(),
    ):
        report = run_build(config)
    assert report["tables_failed"] == 0, report["failed_tables"]
    with driver.session() as session:
        rows = session.run(
            "SHOW VECTOR INDEXES YIELD name, labelsOrTypes RETURN name, labelsOrTypes",
        ).data()
    return {row["name"]: row["labelsOrTypes"][0] for row in rows}


def _identity(node: dict[str, Any]) -> tuple[tuple[bool, str], ...]:
    return tuple((node.get(key) is None, str(node.get(key) or "")) for key in IDENTITY)


def _exact_top_k(
    driver: Any, label: str, query: list[float], top_k: int,
) -> list[tuple[tuple[bool, str], ...]]:
    with driver.session() as session:
        nodes = session.run(
            f"MATCH (n:`{label}`) WHERE n.embedding IS NOT NULL RETURN properties(n) AS node",
        ).data()
    norm = math.sqrt(sum(q * q for q in query))

    def score(node: dict[str, Any]) -> float:
        vector = node["embedding"]
        dot = sum(a * b for a, b in zip(vector, query))
        return (1.0 + dot / (norm * math.sqrt(sum(v * v for v in vector)))) / 2.0

    ranked = sorted((row["node"] for row in nodes), key=lambda n: (-score(n), _identity(n)))
    return [_identity(n) for n in ranked[:top_k]]


def test_each_vector_read_is_the_exact_top_k_under_a_total_order(
    clean_neo4j: Any, tmp_path: Path,
) -> None:
    labels = _build(clean_neo4j, tmp_path)
    engine = RetrievalEngine(driver=clean_neo4j, embedder=HashEmbedder())
    probes = [table.name for table in MAIN_TABLES] + [ASKED_HIERARCHY_QUESTION]
    assert labels, "the build created no vector index"

    differing = []
    for probe in probes:
        query = HashEmbedder().vector(probe)
        for top_k in (3, 10):
            read = engine.vector_search(probe, top_k)
            for index, label in labels.items():
                got = [_identity(c) for c in read if c["index"] == index]
                if got != _exact_top_k(clean_neo4j, label, query, top_k):
                    differing.append((probe, top_k, index))

    assert differing == [], f"{len(differing)} of {len(probes) * 2 * len(labels)} reads differ"
