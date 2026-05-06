"""Source-agnostic metadata-tier classifier.

Classifies a table into `rich` / `sparse` / `name_only` from its L1
evidence shape — never from schema names, source types, or per-source
allowlists. Pure: no I/O, no LLM, no side effects.
"""
from __future__ import annotations

from typing import Any, Literal

Tier = Literal["rich", "sparse", "name_only"]

_TOP_VALUES_MAJORITY_FLOOR = 0.50
_DEFAULT_RICH_COMMENT_FLOOR = 0.60


def _column_comment_coverage(columns: list[dict[str, Any]]) -> float:
    if not columns:
        return 0.0
    with_comments = sum(
        1 for c in columns
        if (c.get("column_comment") or c.get("comment") or "").strip()
    )
    return with_comments / len(columns)


def _top_values_coverage(columns: list[dict[str, Any]]) -> float:
    if not columns:
        return 0.0
    with_top = sum(
        1 for c in columns if c.get("top_values")
    )
    return with_top / len(columns)


def classify_metadata_tier(
    evidence: dict[str, Any],
    *,
    rich_floor: float = _DEFAULT_RICH_COMMENT_FLOOR,
) -> Tier:
    columns = evidence.get("columns") or []
    table_comment = (
        evidence.get("table_comment")
        or evidence.get("comment")
        or ""
    ).strip()
    sample_rows = evidence.get("sample_rows") or []

    comment_cov = _column_comment_coverage(columns)
    top_cov = _top_values_coverage(columns)
    has_supporting = (
        bool(table_comment)
        or top_cov >= _TOP_VALUES_MAJORITY_FLOOR
        or bool(sample_rows)
    )

    if comment_cov >= rich_floor and has_supporting:
        return "rich"

    has_any_evidence = (
        comment_cov > 0
        or bool(table_comment)
        or top_cov > 0
        or bool(sample_rows)
    )
    if has_any_evidence:
        return "sparse"
    return "name_only"
