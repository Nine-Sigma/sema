"""Neo4j vector-index helpers used by the embedding dimension guard.

Used in two places:
  - write path (`_compute_embeddings`): verify embedder matches existing
    index before any vectors are written.
  - retrieval path (`RetrievalEngine.__init__`): verify embedder matches
    existing indexes before retrieval runs, so a provider swap cannot
    silently degrade to lexical-only.
"""
from __future__ import annotations

from typing import Any


class EmbeddingDimensionMismatchError(RuntimeError):
    """Raised when a vector-index's declared dimension differs from the embedder.

    Narrow subclass of RuntimeError so write- and retrieval-path callers
    can re-raise it past broad `except Exception` degradations.
    """


_SHOW_VECTOR_INDEX_QUERY = (
    "SHOW VECTOR INDEXES YIELD name, options "
    "WHERE name = $name "
    "RETURN options AS options"
)


def get_declared_index_dimension(
    driver: Any, index_name: str,
) -> int | None:
    """Return the declared `vector.dimensions` of a vector index, or None.

    Defensive: returns None on any driver / session / record error or on
    unexpected shapes, so a broken Neo4j state cannot crash retrieval
    startup. The guard can only raise on a *confirmed* mismatch.
    """
    try:
        with driver.session() as session:
            records = session.run(_SHOW_VECTOR_INDEX_QUERY, name=index_name)
            for record in records:
                options = _record_options(record)
                dim = _extract_declared_dim(options)
                if dim is not None:
                    return dim
    except Exception:
        return None
    return None


def _record_options(record: Any) -> Any:
    if hasattr(record, "get"):
        return record.get("options")
    try:
        return record["options"]
    except Exception:
        return None


def _extract_declared_dim(options: Any) -> int | None:
    if not isinstance(options, dict):
        return None
    index_config = options.get("indexConfig") or options.get("indexconfig")
    if not isinstance(index_config, dict):
        return None
    dim = index_config.get("vector.dimensions")
    try:
        return int(dim) if dim is not None else None
    except (TypeError, ValueError):
        return None


def probe_embedder_dimension(embedder: Any) -> int:
    """Measure an embedder's output dimension via one trivial call."""
    if hasattr(embedder, "get_sentence_embedding_dimension"):
        return int(embedder.get_sentence_embedding_dimension())
    if hasattr(embedder, "embed_query"):
        vec = embedder.embed_query("dimension probe")
        return len(vec)
    if hasattr(embedder, "embed_documents"):
        return len(embedder.embed_documents(["dimension probe"])[0])
    if hasattr(embedder, "encode"):
        return len(embedder.encode(["dimension probe"])[0])
    raise ValueError(
        "Embedder exposes no known probe interface "
        "(embed_query / embed_documents / encode / "
        "get_sentence_embedding_dimension).",
    )


def assert_write_dim_matches(
    driver: Any,
    index_names: list[str],
    embedder_dim: int,
    *,
    model_name: str,
) -> None:
    """Raise if any existing index's dim disagrees with embedder_dim (write path)."""
    for name in index_names:
        existing = get_declared_index_dimension(driver, name)
        if existing is None or existing == embedder_dim:
            continue
        raise EmbeddingDimensionMismatchError(
            _dim_mismatch_message(
                context="write",
                model_name=model_name,
                embedder_dim=embedder_dim,
                index_name=name,
                existing_dim=existing,
            )
        )


def assert_retrieval_dim_matches(
    driver: Any,
    embedder: Any,
    index_names: list[str],
    *,
    model_name: str,
) -> None:
    """Raise if any existing index's dim disagrees with the embedder (retrieval path).

    Probes the embedder only if at least one index exists — keeps
    no-index tests fast and mock-friendly.
    """
    raw_dims = {
        name: get_declared_index_dimension(driver, name)
        for name in index_names
    }
    existing_by_name: dict[str, int] = {
        n: d for n, d in raw_dims.items() if d is not None
    }
    if not existing_by_name:
        return
    embedder_dim = probe_embedder_dimension(embedder)
    for name, existing_dim in existing_by_name.items():
        if existing_dim != embedder_dim:
            raise EmbeddingDimensionMismatchError(
                _dim_mismatch_message(
                    context="retrieval",
                    model_name=model_name,
                    embedder_dim=embedder_dim,
                    index_name=name,
                    existing_dim=existing_dim,
                )
            )


def _dim_mismatch_message(
    *,
    context: str,
    model_name: str,
    embedder_dim: int,
    index_name: str,
    existing_dim: int,
) -> str:
    surface_hint = (
        "Affects `sema context` / `sema query` startup."
        if context == "retrieval"
        else "Affects `sema build` before any vectors are written."
    )
    return (
        f"Embedding dimension mismatch ({context} path): model "
        f"'{model_name}' produces {embedder_dim}-dim vectors but existing "
        f"index '{index_name}' declares {existing_dim}-dim. {surface_hint} "
        f"Re-embed the graph with a matching model or drop the index "
        f"before re-building."
    )
