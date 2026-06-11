"""SQL prompt construction from an SCO."""

from __future__ import annotations

from sema.log import logger
from sema.consumers.nl2sql.prompting_utils import (
    _add_ancestry_section,
    _add_dialect_notes,
    _add_entity_context_section,
    _add_governed_values_section,
    _add_join_section,
    _add_metrics_section,
    _add_rules_and_question,
    _add_tables_and_columns,
    _dialect_label,
)
from sema.models.context import SemanticContextObject


def build_sql_prompt(
    sco: SemanticContextObject,
    question: str,
    dialect: str = "databricks",
    max_chars: int = 12000,
) -> str:
    """Build constrained SQL generation prompt from an SCO.

    If the prompt exceeds max_chars, sections are dropped in a
    strict deterministic cut order until it fits.
    """
    parts = _build_prompt_parts(sco, question, dialect)
    prompt = "\n".join(parts)

    if len(prompt) <= max_chars:
        return prompt

    logger.warning(
        "Prompt exceeds budget ({} > {}), truncating",
        len(prompt), max_chars,
    )
    return _truncate_prompt(sco, question, dialect, max_chars)


def _build_prompt_parts(
    sco: SemanticContextObject,
    question: str,
    dialect: str = "databricks",
) -> list[str]:
    """Build all prompt sections without truncation."""
    parts: list[str] = [
        f"You are a SQL expert for {_dialect_label(dialect)}. "
        "Generate a SQL query using ONLY the tables, columns, "
        "and values provided below. "
        "Do not use any table or column not listed.\n",
    ]
    _add_entity_context_section(parts, sco)
    _add_tables_and_columns(parts, sco)
    _add_join_section(parts, sco)
    _add_governed_values_section(parts, sco)
    _add_metrics_section(parts, sco)
    _add_ancestry_section(parts, sco)
    _add_dialect_notes(parts, dialect)
    _add_rules_and_question(parts, question)
    return parts


def _truncate_prompt(
    sco: SemanticContextObject,
    question: str,
    dialect: str,
    max_chars: int,
) -> str:
    """Truncate in strict order until prompt fits budget.

    Cut order: (1) governed values, (2) hierarchy, (3) semantic
    annotations, (4) metrics, (5) column lists, (6) dialect notes.
    """
    # Cut 1: Truncate governed values to 5 per column
    prompt = _build_with_cuts(
        sco, question, dialect,
        max_values=5, annotate=True,
    )
    if len(prompt) <= max_chars:
        return prompt

    # Cut 2: Remove governed values entirely
    prompt = _build_with_cuts(
        sco, question, dialect,
        skip_values=True, annotate=True,
    )
    if len(prompt) <= max_chars:
        return prompt

    # Cut 3: Remove hierarchy
    prompt = _build_with_cuts(
        sco, question, dialect,
        skip_values=True, skip_ancestry=True,
        annotate=True,
    )
    if len(prompt) <= max_chars:
        return prompt

    # Cut 4: Remove semantic annotations
    prompt = _build_with_cuts(
        sco, question, dialect,
        skip_values=True, skip_ancestry=True,
        annotate=False,
    )
    if len(prompt) <= max_chars:
        return prompt

    # Cut 5: Remove metrics
    prompt = _build_with_cuts(
        sco, question, dialect,
        skip_values=True, skip_ancestry=True,
        skip_metrics=True, annotate=False,
    )
    if len(prompt) <= max_chars:
        return prompt

    # Cut 6: Truncate column lists to 10
    prompt = _build_with_cuts(
        sco, question, dialect,
        skip_values=True, skip_ancestry=True,
        skip_metrics=True, annotate=False,
        max_cols=10,
    )
    if len(prompt) <= max_chars:
        return prompt

    # Cut 7: Remove dialect notes
    prompt = _build_with_cuts(
        sco, question, dialect,
        skip_values=True, skip_ancestry=True,
        skip_metrics=True, skip_dialect=True,
        annotate=False, max_cols=10,
    )
    return prompt


def _build_with_cuts(
    sco: SemanticContextObject,
    question: str,
    dialect: str,
    max_values: int | None = None,
    max_cols: int | None = None,
    annotate: bool = True,
    skip_values: bool = False,
    skip_ancestry: bool = False,
    skip_metrics: bool = False,
    skip_dialect: bool = False,
) -> str:
    """Build prompt with selective section omission."""
    parts: list[str] = [
        f"You are a SQL expert for {_dialect_label(dialect)}. "
        "Generate a SQL query using ONLY the tables, columns, "
        "and values provided below. "
        "Do not use any table or column not listed.\n",
    ]
    _add_entity_context_section(parts, sco)
    _add_tables_and_columns(
        parts, sco, max_cols=max_cols, annotate=annotate,
    )
    _add_join_section(parts, sco)
    if not skip_values:
        _add_governed_values_section(
            parts, sco, max_values=max_values,
        )
    if not skip_metrics:
        _add_metrics_section(parts, sco)
    if not skip_ancestry:
        _add_ancestry_section(parts, sco)
    if not skip_dialect:
        _add_dialect_notes(parts, dialect)
    _add_rules_and_question(parts, question)
    return "\n".join(parts)
