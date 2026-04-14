"""Structured diff between assertion dumps."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Predicates whose removal is a regression risk
_HIGH_VALUE_PREDICATES = frozenset({
    "has_entity_name",
    "has_property_name",
    "has_semantic_type",
})


def diff_dumps(
    dump_a: dict[str, Any],
    dump_b: dict[str, Any],
) -> dict[str, Any]:
    """Compare two assertion dumps, report added/removed/changed.

    Assertions are keyed by (subject_ref, predicate). Source field
    changes alone are ignored — only payload and confidence are semantic.
    """
    index_a = _build_index(dump_a["assertions"])
    index_b = _build_index(dump_b["assertions"])

    keys_a = set(index_a.keys())
    keys_b = set(index_b.keys())

    added = [
        index_b[k] for k in sorted(keys_b - keys_a)
    ]
    removed = [
        _tag_regression(index_a[k]) for k in sorted(keys_a - keys_b)
    ]
    changed = _find_changes(index_a, index_b, keys_a & keys_b)

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "summary": {
            "added_count": len(added),
            "removed_count": len(removed),
            "changed_count": len(changed),
            "total_before": len(dump_a["assertions"]),
            "total_after": len(dump_b["assertions"]),
        },
    }


def diff_dump_files(
    path_a: Path,
    path_b: Path,
) -> dict[str, Any]:
    """Diff two assertion dump files on disk."""
    dump_a = json.loads(path_a.read_text())
    dump_b = json.loads(path_b.read_text())
    return diff_dumps(dump_a, dump_b)


def _build_index(
    assertions: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index assertions by (subject_ref, predicate)."""
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for a in assertions:
        key = (a["subject_ref"], a["predicate"])
        index[key] = a
    return index


def _find_changes(
    index_a: dict[tuple[str, str], dict[str, Any]],
    index_b: dict[tuple[str, str], dict[str, Any]],
    shared_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Find semantic changes (payload or confidence) in shared keys."""
    changes: list[dict[str, Any]] = []
    for key in sorted(shared_keys):
        old = index_a[key]
        new = index_b[key]
        payload_changed = old["payload"] != new["payload"]
        confidence_changed = old["confidence"] != new["confidence"]
        if payload_changed or confidence_changed:
            changes.append({
                "subject_ref": key[0],
                "predicate": key[1],
                "old_payload": old["payload"],
                "new_payload": new["payload"],
                "old_confidence": old["confidence"],
                "new_confidence": new["confidence"],
            })
    return changes


def _tag_regression(assertion: dict[str, Any]) -> dict[str, Any]:
    """Tag removed assertions that are regression risks."""
    tagged = dict(assertion)
    tagged["regression_risk"] = (
        assertion["predicate"] in _HIGH_VALUE_PREDICATES
    )
    return tagged
