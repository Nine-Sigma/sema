"""SQL prompt construction from an SCO."""

from __future__ import annotations

from typing import Any

from sema.log import logger
from sema.models.context import SemanticContextObject


def build_sql_prompt(
    sco: SemanticContextObject,
    question: str,
    max_chars: int = 12000,
) -> str:
    """Build constrained SQL generation prompt from an SCO.

    If the prompt exceeds max_chars, governed values are truncated
    first, then column lists, with [truncated] markers.
    """
    parts = _build_prompt_parts(sco, question)
    prompt = "\n".join(parts)

    if len(prompt) <= max_chars:
        return prompt

    logger.warning(
        "Prompt exceeds budget ({} > {}), truncating",
        len(prompt), max_chars,
    )
    return _truncate_prompt(sco, question, max_chars)


def _build_prompt_parts(
    sco: SemanticContextObject, question: str,
) -> list[str]:
    """Build all prompt sections without truncation."""
    parts = [
        "You are a SQL expert for Databricks. Generate a SQL query "
        "using ONLY the tables, columns, and values provided below. "
        "Do not use any table or column not listed.\n",
    ]
    parts.append("AVAILABLE TABLES AND COLUMNS:")
    for asset in sco.physical_assets:
        fqn = f"{asset.catalog}.{asset.schema}.{asset.table}"
        parts.append(f"  {fqn}: {', '.join(asset.columns)}")

    _add_join_section(parts, sco)
    _add_governed_values_section(parts, sco)
    _add_rules_and_question(parts, question)
    return parts


def _add_join_section(
    parts: list[str], sco: SemanticContextObject,
) -> None:
    if not sco.join_paths:
        return
    parts.append("\nJOIN PATHS:")
    for jp in sco.join_paths:
        predicates_str = _format_join_predicates(jp)
        cardinality = jp.cardinality_hint or "unknown"
        parts.append(
            f"  {jp.from_table} -> {jp.to_table} "
            f"ON {predicates_str} ({cardinality})"
        )


def _format_join_predicates(jp: Any) -> str:
    """Format join predicates from a JoinPath."""
    if jp.sql_snippet:
        return str(jp.sql_snippet)
    if jp.join_predicates:
        return " AND ".join(
            f"{p.left_table}.{p.left_column} {p.operator} "
            f"{p.right_table}.{p.right_column}"
            for p in jp.join_predicates
        )
    return "(unknown join condition)"


def _add_governed_values_section(
    parts: list[str],
    sco: SemanticContextObject,
    max_values: int | None = None,
) -> None:
    if not sco.governed_values:
        return
    parts.append("\nGOVERNED FILTER VALUES (use these exact values):")
    for gv in sco.governed_values:
        values = gv.values
        if max_values is not None and len(values) > max_values:
            values = values[:max_values]
            truncated = True
        else:
            truncated = False
        values_str = ", ".join(f"'{v['code']}'" for v in values)
        suffix = " [truncated]" if truncated else ""
        parts.append(
            f"  {gv.table}.{gv.column}: [{values_str}]{suffix}"
        )


def _add_rules_and_question(
    parts: list[str], question: str,
) -> None:
    parts.append("\nRULES:")
    parts.append(
        "- Use ONLY the tables and columns listed above"
    )
    parts.append(
        "- Use ONLY the filter values provided — "
        "do not guess or abbreviate"
    )
    parts.append("- Join tables using the join paths provided")
    parts.append(
        "- Column names are exact — copy them precisely as shown"
    )
    parts.append(
        "- Use fully qualified table names (catalog.schema.table)"
    )
    parts.append(f"\nQuestion: {question}")
    parts.append(
        "\nReturn ONLY the SQL query, no markdown or explanation."
    )


def _truncate_prompt(
    sco: SemanticContextObject,
    question: str,
    max_chars: int,
) -> str:
    """Truncate governed values, then columns to fit budget."""
    parts = [
        "You are a SQL expert for Databricks. Generate a SQL query "
        "using ONLY the tables, columns, and values provided below. "
        "Do not use any table or column not listed.\n",
    ]

    parts.append("AVAILABLE TABLES AND COLUMNS:")
    for asset in sco.physical_assets:
        fqn = f"{asset.catalog}.{asset.schema}.{asset.table}"
        parts.append(f"  {fqn}: {', '.join(asset.columns)}")

    _add_join_section(parts, sco)
    _add_governed_values_section(parts, sco, max_values=5)
    _add_rules_and_question(parts, question)

    prompt = "\n".join(parts)
    if len(prompt) <= max_chars:
        return prompt

    # Further truncation: trim column lists
    parts_trimmed = [
        "You are a SQL expert for Databricks. Generate a SQL query "
        "using ONLY the tables, columns, and values provided below. "
        "Do not use any table or column not listed.\n",
    ]
    parts_trimmed.append("AVAILABLE TABLES AND COLUMNS:")
    for asset in sco.physical_assets:
        fqn = f"{asset.catalog}.{asset.schema}.{asset.table}"
        cols = asset.columns[:10]
        suffix = " [truncated]" if len(asset.columns) > 10 else ""
        parts_trimmed.append(
            f"  {fqn}: {', '.join(cols)}{suffix}"
        )
    _add_join_section(parts_trimmed, sco)
    _add_governed_values_section(parts_trimmed, sco, max_values=3)
    _add_rules_and_question(parts_trimmed, question)

    return "\n".join(parts_trimmed)
