"""Helpers that extract artifact fields from a failure exception."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
from typing import Any


def _safe_serialize(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            return None
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return asdict(value)
        except Exception:
            return None
    return value


def _stage_b_from_exc(exc: BaseException) -> Any:
    return getattr(exc, "stage_b", None)


def _stage_a_from_exc(exc: BaseException) -> Any:
    return getattr(exc, "stage_a", None)


def stage_a_output_for_exc(exc: BaseException) -> Any:
    return _safe_serialize(_stage_a_from_exc(exc))


def unresolved_columns_for_exc(exc: BaseException) -> list[dict[str, Any]]:
    sb = _stage_b_from_exc(exc)
    if sb is None or not getattr(sb, "unresolved_columns", None):
        return []
    out = []
    for u in sb.unresolved_columns:
        ud = _safe_serialize(u)
        if isinstance(ud, dict):
            out.append(ud)
    return out


def counters_for_exc(exc: BaseException) -> dict[str, int]:
    sb = _stage_b_from_exc(exc)
    if sb is None:
        return {
            "batches_attempted": 0,
            "batches_succeeded": 0,
            "retries_used": 0,
            "splits_used": 0,
            "rescues_used": 0,
        }
    batches = list(getattr(sb, "batch_results", []) or [])
    return {
        "batches_attempted": len(batches),
        "batches_succeeded": len(batches),
        "retries_used": int(getattr(sb, "retries_used", 0) or 0),
        "splits_used": int(getattr(sb, "splits_used", 0) or 0),
        "rescues_used": int(getattr(sb, "rescues_used", 0) or 0),
    }


def llm_attempts_for_exc(exc: BaseException) -> list[dict[str, Any]]:
    attempts = getattr(exc, "llm_attempts", []) or []
    out = []
    for a in attempts:
        d = _safe_serialize(a)
        if isinstance(d, dict):
            out.append(d)
    return out


def step_errors_for_exc(exc: BaseException) -> list[dict[str, str]]:
    step_errors = getattr(exc, "step_errors", []) or []
    return [
        {
            "step_name": name,
            "exception_type": type(err).__name__,
            "message": str(err),
        }
        for name, err in step_errors
    ]


def _hash_strings(parts: list[str]) -> str:
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def prompt_hashes_for_exc(exc: BaseException) -> dict[str, str | None]:
    by_stage: dict[str, list[str]] = {}
    for a in getattr(exc, "llm_attempts", []) or []:
        stage = getattr(a, "stage", None)
        prompt = getattr(a, "prompt_text", None)
        if not stage or not prompt:
            continue
        by_stage.setdefault(stage, []).append(prompt)
    return {
        "stage_a": _hash_strings(by_stage["L2 stage_a"])
        if "L2 stage_a" in by_stage else None,
        "stage_b": _hash_strings(by_stage["L2 stage_b"])
        if "L2 stage_b" in by_stage else None,
        "stage_c": _hash_strings(by_stage["L2 stage_c"])
        if "L2 stage_c" in by_stage else None,
    }
