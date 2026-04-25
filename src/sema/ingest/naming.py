from __future__ import annotations

import hashlib
import re

_MAX_IDENT_LEN = 63
_HASH_LEN = 10
_NON_IDENT_CHAR = re.compile(r"[^a-z0-9_]")
_RUN_OF_UNDERSCORES = re.compile(r"_+")


def sanitize_schema_name(prefix: str, study_id: str) -> str:
    if not study_id:
        raise ValueError("study_id must be non-empty")

    sanitized = _sanitize_identifier(study_id)
    if not sanitized:
        raise ValueError(f"study_id {study_id!r} produces empty identifier after sanitization")

    full = f"{prefix}_{sanitized}"
    if len(full) <= _MAX_IDENT_LEN:
        return full

    return _truncate_with_hash(prefix, sanitized, study_id)


def _sanitize_identifier(raw: str) -> str:
    lowered = raw.lower()
    replaced = _NON_IDENT_CHAR.sub("_", lowered)
    collapsed = _RUN_OF_UNDERSCORES.sub("_", replaced)
    return collapsed.strip("_")


def _truncate_with_hash(prefix: str, sanitized: str, original: str) -> str:
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:_HASH_LEN]
    suffix = f"_{digest}"
    available = _MAX_IDENT_LEN - len(prefix) - 1 - len(suffix)
    if available <= 0:
        raise ValueError(
            f"prefix {prefix!r} leaves no room for sanitized study_id within {_MAX_IDENT_LEN} chars"
        )
    truncated = sanitized[:available].rstrip("_")
    if not truncated:
        raise ValueError(
            f"sanitized study_id collapses to empty after truncation for prefix {prefix!r}"
        )
    return f"{prefix}_{truncated}{suffix}"
