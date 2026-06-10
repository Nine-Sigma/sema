"""Canonical-JSON serialization helpers used by SnapshotHasher and card hashing.

The canonical form sorts keys, omits whitespace separators, renders floats via
`repr()` against IEEE-754, and renders datetimes as ISO 8601 with a `Z` suffix.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any


def normalize_for_canonical_json(value: Any) -> Any:
    if isinstance(value, datetime):
        return _iso_with_z(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): normalize_for_canonical_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_for_canonical_json(item) for item in value]
    if isinstance(value, float):
        return _CanonicalFloat(value)
    return value


def _iso_with_z(value: datetime) -> str:
    iso = value.isoformat()
    if value.tzinfo is None:
        return iso + "Z"
    if iso.endswith("+00:00"):
        return iso[: -len("+00:00")] + "Z"
    return iso


class _CanonicalFloat(float):
    """Float subclass that renders via `repr()` for canonical-JSON stability."""

    def __repr__(self) -> str:
        return repr(float(self))


def _default(obj: Any) -> Any:  # pragma: no cover - dispatched only for non-stdlib types
    raise TypeError(f"Type {type(obj).__name__} is not JSON-serializable")


def canonical_dumps(value: Any) -> str:
    normalized = normalize_for_canonical_json(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_default,
        allow_nan=False,
    )
