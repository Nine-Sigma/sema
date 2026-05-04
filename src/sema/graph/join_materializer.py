"""Join path materialization helpers.

Normalizes JOINS_TO + HAS_JOIN_EVIDENCE + FK_TO into JoinPath nodes.
Extracted from materializer_utils.py to keep files under 400 lines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sema.graph.join_materializer_utils import normalize_fk_to_assertion
from sema.graph.loader_utils import batch_upsert_join_paths
from sema.models.assertions import Assertion, AssertionPredicate

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader

_JOIN_PREDICATES = (
    AssertionPredicate.HAS_JOIN_EVIDENCE.value,
    AssertionPredicate.JOINS_TO.value,
    AssertionPredicate.FK_TO.value,
)


def _derive_join_path_name(
    join_predicates: list[dict[str, str]],
) -> str:
    parts = []
    for jp in join_predicates:
        left = f"{jp['left_table']}/{jp['left_column']}"
        right = f"{jp['right_table']}/{jp['right_column']}"
        parts.append(f"{left}={right}")
    return "|".join(parts)


def _build_join_path_records(
    join_groups: dict[str, list[Assertion]],
) -> list[dict[str, Any]]:
    from sema.graph.materializer_utils import pick_winner

    records: list[dict[str, Any]] = []
    for _subject_ref, group in join_groups.items():
        winner = pick_winner(group)
        if not winner:
            continue
        join_predicates = winner.payload.get("join_predicates", [])
        hop_count = winner.payload.get("hop_count", 1)
        cardinality = winner.payload.get("cardinality")
        name = _derive_join_path_name(join_predicates)
        records.append({
            "name": name,
            "join_predicates": join_predicates,
            "hop_count": hop_count,
            "source": winner.source,
            "confidence": winner.confidence,
            "sql_snippet": winner.payload.get("sql_snippet"),
            "cardinality_hint": cardinality,
            "from_table": winner.payload.get("from_table", ""),
            "to_table": winner.payload.get("to_table", ""),
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
        name = rec["name"]
        for jp in rec["join_predicates"]:
            if jp.get("left_table"):
                loader.add_join_path_uses(
                    name, jp["left_table"],
                    source_schema=source_schema,
                )
            if jp.get("left_column") and jp.get("left_table"):
                loader.add_join_path_uses(
                    name, jp["left_table"], jp["left_column"],
                    source_schema=source_schema,
                )
            if jp.get("right_table"):
                loader.add_join_path_uses(
                    name, jp["right_table"],
                    source_schema=source_schema,
                )
            if jp.get("right_column") and jp.get("right_table"):
                loader.add_join_path_uses(
                    name, jp["right_table"], jp["right_column"],
                    source_schema=source_schema,
                )
        if rec.get("from_table") or rec.get("to_table"):
            loader.add_join_path_entity_links(
                name, rec.get("from_table", ""), rec.get("to_table", ""),
                source_schema=source_schema,
            )


def materialize_join_paths(
    loader: GraphLoader,
    groups: dict[tuple[str, str], list[Assertion]],
    source_schema: str | None = None,
) -> None:
    """Normalize JOINS_TO + HAS_JOIN_EVIDENCE + FK_TO into JoinPath nodes."""
    join_groups: dict[str, list[Assertion]] = {}
    for (subj, pred), group in groups.items():
        if pred not in _JOIN_PREDICATES:
            continue
        normalized = (
            [normalize_fk_to_assertion(a) for a in group]
            if pred == AssertionPredicate.FK_TO.value
            else list(group)
        )
        if subj in join_groups:
            join_groups[subj].extend(normalized)
        else:
            join_groups[subj] = normalized
    records = _build_join_path_records(join_groups)
    batch = [
        {k: v for k, v in r.items()
         if k not in ("from_table", "to_table")}
        for r in records
    ]
    batch_upsert_join_paths(loader, batch, source_schema=source_schema)
    _wire_join_path_edges(loader, records, source_schema)
