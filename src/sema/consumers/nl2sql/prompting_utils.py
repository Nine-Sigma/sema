"""Prompt section builders for the NL2SQL consumer.

Extracted from prompting.py to keep that module focused on the
build_sql_prompt entry point and its truncation budget.
"""

from __future__ import annotations

from typing import Any

from sema.models.context import (
    ResolvedProperty,
    SemanticContextObject,
)


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


def _format_values_line(
    values: list[dict[str, str]], max_values: int | None,
) -> str:
    truncated = max_values is not None and len(values) > max_values
    if truncated:
        values = values[:max_values]
    values_str = ", ".join(f"'{v['code']}'" for v in values)
    suffix = " [truncated]" if truncated else ""
    return f"[{values_str}]{suffix}"


def _add_governed_values_section(
    parts: list[str],
    sco: SemanticContextObject,
    max_values: int | None = None,
) -> None:
    authoritative = [
        gv for gv in sco.governed_values if not gv.ambiguous
    ]
    if authoritative:
        parts.append(
            "\nGOVERNED FILTER VALUES (use these exact values):"
        )
        for gv in authoritative:
            line = _format_values_line(gv.values, max_values)
            parts.append(f"  {gv.table}.{gv.column}: {line}")
    _add_ambiguous_values_section(parts, sco, max_values)


def _add_ambiguous_values_section(
    parts: list[str],
    sco: SemanticContextObject,
    max_values: int | None = None,
) -> None:
    ambiguous = [gv for gv in sco.governed_values if gv.ambiguous]
    if not ambiguous:
        return
    parts.append(
        "\nAMBIGUOUS CODES (the same code exists in multiple "
        "vocabularies — use ONLY the column matching the question "
        "intent):"
    )
    for gv in ambiguous:
        line = _format_values_line(gv.values, max_values)
        vocab = (
            f" (vocabulary: {gv.vocabulary})" if gv.vocabulary else ""
        )
        parts.append(f"  {gv.table}.{gv.column}: {line}{vocab}")


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
