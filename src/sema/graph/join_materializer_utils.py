"""Helpers for `join_materializer.py` — kept thin per the engine-file rule."""
from __future__ import annotations

from sema.models.assertions import Assertion, AssertionPredicate


def _strip_trailing_segment(ref: str) -> str:
    return ref.rsplit("/", 1)[0] if ref else ""


def normalize_fk_to_assertion(a: Assertion) -> Assertion:
    """Translate an `FK_TO` assertion into the legacy join-evidence shape.

    The legacy materializer pipeline reads `join_predicates` /
    `from_table` / `to_table` from the winner's payload. FK_TO emits
    `pk_table` / `pk_column` / `fk_table` / `fk_column` with subject_ref
    pointing at the FK column and object_ref at the PK column. This
    helper translates one shape into the other so a single materializer
    path serves both predicate families.
    """
    p = a.payload
    pk_table, pk_column = p["pk_table"], p["pk_column"]
    fk_table, fk_column = p["fk_table"], p["fk_column"]
    new_payload = {
        "join_predicates": [{
            "left_table": pk_table, "left_column": pk_column,
            "right_table": fk_table, "right_column": fk_column,
            "operator": "=",
        }],
        "hop_count": 1,
        "from_table": _strip_trailing_segment(a.object_ref or ""),
        "to_table": _strip_trailing_segment(a.subject_ref),
        "tier": p.get("tier"),
    }
    return a.model_copy(update={
        "payload": new_payload,
        "predicate": AssertionPredicate.HAS_JOIN_EVIDENCE,
    })
