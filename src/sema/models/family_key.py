"""Assertion family key for cross-run correction matching.

A family key identifies the *kind* of claim an assertion makes,
independent of run-specific fields like assertion ID, confidence,
or timestamp. Two assertions with the same family key are competing
claims about the same thing.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Final

from sema.models.assertions import AssertionPredicate


def _payload_value(payload: dict[str, Any]) -> str | None:
    return payload.get("value")


def _payload_code(payload: dict[str, Any]) -> str | None:
    return payload.get("code")


def _payload_none(payload: dict[str, Any]) -> str | None:
    return None


PAYLOAD_IDENTITY_EXTRACTORS: Final[dict[
    AssertionPredicate, Callable[[dict[str, Any]], str | None]
]] = {
    AssertionPredicate.VOCABULARY_MATCH: _payload_value,
    AssertionPredicate.HAS_ENTITY_NAME: _payload_value,
    AssertionPredicate.HAS_PROPERTY_NAME: _payload_value,
    AssertionPredicate.HAS_SEMANTIC_TYPE: _payload_value,
    AssertionPredicate.HAS_DECODED_VALUE: _payload_code,
    AssertionPredicate.HAS_ALIAS: _payload_value,
    AssertionPredicate.HAS_SYNONYM: _payload_value,
    AssertionPredicate.PARENT_OF: _payload_none,
    AssertionPredicate.HAS_JOIN_EVIDENCE: _payload_none,
    AssertionPredicate.JOINS_TO: _payload_none,
    AssertionPredicate.TABLE_EXISTS: _payload_none,
    AssertionPredicate.COLUMN_EXISTS: _payload_none,
    AssertionPredicate.HAS_DATATYPE: _payload_value,
    AssertionPredicate.HAS_LABEL: _payload_value,
    AssertionPredicate.HAS_DESCRIPTION: _payload_value,
    AssertionPredicate.HAS_COMMENT: _payload_value,
    AssertionPredicate.HAS_TAG: _payload_value,
    AssertionPredicate.HAS_TOP_VALUES: _payload_none,
    AssertionPredicate.HAS_SAMPLE_ROWS: _payload_none,
    AssertionPredicate.MAPS_TO: _payload_value,
    AssertionPredicate.ENTITY_ON_TABLE: _payload_none,
    AssertionPredicate.PROPERTY_ON_COLUMN: _payload_none,
}


def payload_identity(
    predicate: AssertionPredicate,
    payload: dict[str, Any],
) -> str | None:
    extractor = PAYLOAD_IDENTITY_EXTRACTORS.get(predicate, _payload_none)
    return extractor(payload)


def family_key(
    subject_ref: str,
    predicate: AssertionPredicate,
    payload: dict[str, Any],
    object_ref: str | None = None,
) -> str:
    """Compute a stable family key for cross-run assertion matching.

    Two assertions with the same family key are competing claims
    about the same subject/predicate/identity/object combination.
    """
    pid = payload_identity(predicate, payload)
    parts = (subject_ref, predicate.value, pid or "", object_ref or "")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()
