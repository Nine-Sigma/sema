"""SQL prompt construction from an SCO."""

from __future__ import annotations

from typing import Any

from sema.log import logger
from sema.models.context import (
    ResolvedProperty,
    SemanticContextObject,
)


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


def _dialect_label(dialect: str) -> str:
    if dialect == "databricks":
        return "Databricks"
    return "SQL"


def _add_entity_context_section(
    parts: list[str], sco: SemanticContextObject,
) -> None:
    if not sco.entities:
        return
    has_descriptions = any(e.description for e in sco.entities)
    if not has_descriptions:
        return
    parts.append("ENTITY CONTEXT:")
    for entity in sco.entities:
        if entity.description:
            parts.append(
                f"  {entity.name}: {entity.description}"
            )


def _add_tables_and_columns(
    parts: list[str],
    sco: SemanticContextObject,
    max_cols: int | None = None,
    annotate: bool = True,
) -> None:
    parts.append("\nAVAILABLE TABLES AND COLUMNS:")
    prop_map = _build_property_map(sco) if annotate else {}
    for asset in sco.physical_assets:
        fqn = f"{asset.catalog}.{asset.schema}.{asset.table}"
        cols = asset.columns
        if max_cols is not None and len(cols) > max_cols:
            cols = cols[:max_cols]
            annotated = _annotate_columns(
                cols, fqn, prop_map,
            ) if annotate else cols
            parts.append(
                f"  {fqn}: {', '.join(annotated)} [truncated]"
            )
        else:
            annotated = _annotate_columns(
                cols, fqn, prop_map,
            ) if annotate else cols
            parts.append(f"  {fqn}: {', '.join(annotated)}")


def _build_property_map(
    sco: SemanticContextObject,
) -> dict[tuple[str, str], ResolvedProperty]:
    """Map (physical_table, physical_column) → ResolvedProperty."""
    result: dict[tuple[str, str], ResolvedProperty] = {}
    for entity in sco.entities:
        for prop in entity.properties:
            key = (prop.physical_table, prop.physical_column)
            result[key] = prop
    return result


def _annotate_columns(
    columns: list[str],
    fqn: str,
    prop_map: dict[tuple[str, str], ResolvedProperty],
) -> list[str]:
    """Annotate columns with semantic type where available."""
    result: list[str] = []
    for col in columns:
        prop = prop_map.get((fqn, col))
        if prop and prop.semantic_type != "free_text":
            result.append(f"{col} ({prop.semantic_type})")
        else:
            result.append(col)
    return result


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


def _add_metrics_section(
    parts: list[str], sco: SemanticContextObject,
) -> None:
    if not sco.metrics:
        return
    parts.append("\nMETRIC DEFINITIONS:")
    for m in sco.metrics:
        desc = m.description or ""
        formula = m.formula or ""
        line = f"  {m.name}"
        if desc:
            line += f": {desc}"
        if formula:
            line += f" [{formula}]"
        parts.append(line)


def _add_ancestry_section(
    parts: list[str], sco: SemanticContextObject,
) -> None:
    if not sco.ancestry:
        return
    parts.append("\nTERM HIERARCHY:")
    for term in sco.ancestry:
        if term.parent_code:
            parts.append(
                f"  {term.code} ({term.label}) "
                f"-> parent: {term.parent_code}"
            )
        else:
            parts.append(
                f"  {term.code} ({term.label}) [root]"
            )


def _add_dialect_notes(
    parts: list[str], dialect: str,
) -> None:
    if dialect == "databricks":
        parts.append("\nDIALECT NOTES (Databricks SQL):")
        parts.append(
            "- Use fully qualified names: catalog.schema.table"
        )
        parts.append(
            "- Use backticks for reserved words: `select`, `table`"
        )
        parts.append(
            "- Use TIMESTAMP for date/time comparisons"
        )
        parts.append(
            "- String comparisons are case-sensitive"
        )
    else:
        parts.append("\nDIALECT NOTES (ANSI SQL):")
        parts.append(
            "- Use standard ANSI SQL syntax"
        )
        parts.append(
            "- Use double quotes for identifiers with "
            "reserved words"
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
