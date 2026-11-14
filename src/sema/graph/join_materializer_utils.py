"""Helpers for `join_materializer.py` — kept thin per the engine-file rule."""
from __future__ import annotations

from typing import Any

from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.physical_key import CanonicalRef, PhysicalKey


def _strip_trailing_segment(ref: str) -> str:
    return ref.rsplit("/", 1)[0] if ref else ""


def _side(table: str, key: PhysicalKey, other: PhysicalKey) -> str:
    if key.catalog_or_db != other.catalog_or_db:
        return f"{key.catalog_or_db}.{key.schema or ''}.{table}"
    if key.schema != other.schema:
        return f"{key.schema or ''}.{table}"
    return table


def join_path_name(
    predicates: list[dict[str, Any]], from_key: PhysicalKey, to_key: PhysicalKey,
) -> str:
    """Name a join by its column pairs, qualifying tables only where the sides differ."""
    return "|".join(
        f"{_side(p['left_table'], from_key, to_key)}/{p['left_column']}"
        f"{p.get('operator') or '='}"
        f"{_side(p['right_table'], to_key, from_key)}/{p['right_column']}"
        for p in sorted(predicates, key=lambda p: str(p["left_column"]))
    )


def join_keys(a: Assertion) -> tuple[PhysicalKey, PhysicalKey] | None:
    """The table keys of a join payload's two sides; None for an old-shape payload."""
    from_ref, to_ref = a.payload.get("from_ref"), a.payload.get("to_ref")
    if not from_ref or not to_ref or not a.payload.get("join_predicates"):
        return None
    try:
        return CanonicalRef.parse(from_ref), CanonicalRef.parse(to_ref)
    except ValueError:
        return None


def normalize_fk_to_assertion(a: Assertion) -> Assertion:
    """Translate an `FK_TO` assertion into the join payload the materializer reads.

    `FK_TO` has subject_ref on the referencing column and object_ref on the
    referenced column; the referenced relation goes on the left.
    """
    p = a.payload
    new_payload = {
        "join_predicates": [{
            "left_table": p["pk_table"], "left_column": p["pk_column"],
            "right_table": p["fk_table"], "right_column": p["fk_column"],
            "operator": "=",
        }],
        "hop_count": 1,
        "from_ref": _strip_trailing_segment(a.object_ref or ""),
        "to_ref": _strip_trailing_segment(a.subject_ref),
        "tier": p.get("tier"),
    }
    return a.model_copy(update={
        "payload": new_payload,
        "predicate": AssertionPredicate.HAS_JOIN_EVIDENCE,
    })
