"""Join path materialization helpers.

Normalizes JOINS_TO + HAS_JOIN_EVIDENCE + FK_TO into JoinPath nodes.
Extracted from materializer_utils.py to keep files under 400 lines.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from sema.graph.join_materializer_utils import (
    JOIN_PREDICATES,
    fetch_join_assertions,
    join_keys,
    join_path_name,
    normalize_fk_to_assertion,
)
from sema.graph.loader_utils import batch_upsert_join_paths
from sema.log import logger
from sema.models.assertions import Assertion, AssertionPredicate

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader

_EDGE_ONLY_FIELDS = ("from_key", "to_key")


def _group_by_name(assertions: list[Assertion]) -> dict[str, list[Assertion]]:
    by_name: dict[str, list[Assertion]] = {}
    for a in assertions:
        keys = join_keys(a)
        if keys is None:
            logger.warning(
                "Skipping join assertion {} on {}: payload has no parseable "
                "from_ref/to_ref or no join_predicates (old shape; rebuild to migrate)",
                a.id, a.subject_ref,
            )
            continue
        name = join_path_name(a.payload["join_predicates"], *keys)
        by_name.setdefault(name, []).append(a)
    return by_name


def _build_join_path_records(
    by_name: dict[str, list[Assertion]],
) -> list[dict[str, Any]]:
    from sema.graph.materializer_utils import pick_winner

    records: list[dict[str, Any]] = []
    for name, group in by_name.items():
        winner = pick_winner(group)
        keys = join_keys(winner) if winner else None
        if winner is None or keys is None:
            continue
        records.append({
            "name": name,
            "join_predicates": winner.payload["join_predicates"],
            "hop_count": winner.payload.get("hop_count", 1),
            "source": winner.source,
            "confidence": winner.confidence,
            "sql_snippet": winner.payload.get("sql_snippet"),
            "cardinality_hint": winner.payload.get("cardinality"),
            "from_key": keys[0],
            "to_key": keys[1],
        })
    return records


def _wire_join_path_edges(
    loader: GraphLoader,
    records: list[dict[str, Any]],
    source_schema: str | None,
) -> None:
    if source_schema is None:
        return
    for rec in records:
        name, from_key, to_key = rec["name"], rec["from_key"], rec["to_key"]
        keys = [from_key, to_key]
        for jp in rec["join_predicates"]:
            keys.append(replace(from_key, column=jp["left_column"]))
            keys.append(replace(to_key, column=jp["right_column"]))
        for key in keys:
            loader.add_join_path_uses(name, key, source_schema=source_schema)
        loader.add_join_path_entity_links(
            name, from_key, to_key, source_schema=source_schema,
        )


def materialize_join_paths(
    loader: GraphLoader,
    groups: dict[tuple[str, str], list[Assertion]],
    source_schema: str | None = None,
) -> None:
    """Normalize JOINS_TO + HAS_JOIN_EVIDENCE + FK_TO into JoinPath nodes."""
    joins: list[Assertion] = []
    for (_subject, pred), group in groups.items():
        if pred == AssertionPredicate.FK_TO.value:
            joins.extend(normalize_fk_to_assertion(a) for a in group)
        elif pred in JOIN_PREDICATES:
            joins.extend(group)
    records = _build_join_path_records(_group_by_name(joins))
    batch = [
        {k: v for k, v in r.items() if k not in _EDGE_ONLY_FIELDS}
        for r in records
    ]
    batch_upsert_join_paths(loader, batch, source_schema=source_schema)
    _wire_join_path_edges(loader, records, source_schema)


def materialize_schema_joins(
    loader: GraphLoader, source_schema: str, detected: list[Assertion],
) -> None:
    """Write one schema's joins in one batch, once every table of the run is built.

    The stored declared keys and this run's detected joins meet in one batch, so
    a join name gets one winner whatever order the tables were built in.
    """
    groups: dict[tuple[str, str], list[Assertion]] = {}
    for a in [*fetch_join_assertions(loader, source_schema), *detected]:
        groups.setdefault((a.subject_ref, a.predicate.value), []).append(a)
    materialize_join_paths(loader, groups, source_schema=source_schema)
