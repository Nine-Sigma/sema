"""Prompt section builders for the NL2SQL consumer.

Extracted from prompting.py to keep that module focused on the
build_sql_prompt entry point and its truncation budget.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sema.models.context import (
    JoinPredicate,
    LabelSource,
    ResolvedEntity,
    ResolvedProperty,
    SemanticContextObject,
)
from sema.models.target.identity import EntityIdentity, ResolutionRule


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
    prop_map = _build_property_map(sco)
    for asset in sco.physical_assets:
        fqn = f"{asset.catalog}.{asset.schema}.{asset.table}"
        cols = asset.columns
        if max_cols is not None and len(cols) > max_cols:
            cols = cols[:max_cols]
            annotated = _annotate_columns(cols, fqn, prop_map, semantic=annotate)
            parts.append(
                f"  {fqn}: {', '.join(annotated)} [truncated]"
            )
        else:
            annotated = _annotate_columns(cols, fqn, prop_map, semantic=annotate)
            parts.append(f"  {fqn}: {', '.join(annotated)}")
    _add_column_choice(parts, sco)


def _add_column_choice(
    parts: list[str], sco: SemanticContextObject,
) -> None:
    """One line per column the question names that another column's definition
    competes for: the named column, then its alternatives."""
    alternatives: dict[str, list[str]] = {}
    for m in sco.column_matches:
        alternatives.setdefault(f"{m.table}.{m.column}", []).append(
            f"{m.alternative_table}.{m.alternative_column}"
        )
    parts.extend(
        f"  Column the question names: {named}; alternative: {', '.join(others)}"
        for named, others in alternatives.items()
    )


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
    semantic: bool = True,
) -> list[str]:
    """Annotate columns with semantic type where available, and with each label source."""
    result: list[str] = []
    for col in columns:
        prop = prop_map.get((fqn, col))
        notes = _column_notes(prop, semantic) if prop else []
        result.append(f"{col} ({'; '.join(notes)})" if notes else col)
    return result


def _column_notes(prop: ResolvedProperty, semantic: bool) -> list[str]:
    typed = semantic and prop.semantic_type != "free_text"
    return [*([prop.semantic_type] if typed else []), *map(_label_note, prop.labels)]


def _label_note(label: LabelSource) -> str:
    joined = " AND ".join(map(_format_predicate, label.via))
    return f"label: {label.table}.{label.column}" + (f" on {joined}" if joined else "")


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
        return " AND ".join(map(_format_predicate, jp.join_predicates))
    return "(unknown join condition)"


def _format_predicate(p: JoinPredicate) -> str:
    return f"{p.left_table}.{p.left_column} {p.operator} {p.right_table}.{p.right_column}"


def _add_keys_and_entities_section(
    parts: list[str], sco: SemanticContextObject,
) -> None:
    """Each stated key, entity and shared entity, in words that hold in any SQL dialect."""
    lines = [*_key_lines(sco), *_entity_lines(sco)]
    if not lines:
        return
    parts.append("\nKEYS AND ENTITIES:")
    parts.extend(f"  {line}" for line in lines)


def _key_lines(sco: SemanticContextObject) -> Iterator[str]:
    for asset in sco.physical_assets:
        fqn = f"{asset.catalog}.{asset.schema}.{asset.table}"
        if asset.key_columns:
            yield f"{fqn}: one row per {', '.join(asset.key_columns)}"
        for shared in asset.shared_entities:
            clauses = [f"rows belong to {shared.entity} of {shared.owner_table} "
                       f"through {', '.join(shared.key_columns)}",
                       *_ordering_clauses(shared.entity, shared.identity)]
            yield f"{fqn}: {'; '.join(clauses)}"


def _entity_lines(sco: SemanticContextObject) -> Iterator[str]:
    for entity in sco.entities:
        if entity.identity is not None:
            yield f"{entity.physical_table}: {_entity_sentence(entity, entity.identity)}"


def _entity_sentence(entity: ResolvedEntity, identity: EntityIdentity) -> str:
    name, key = entity.name, ", ".join(entity.key_columns or identity.key_columns)
    clauses = [f"several rows can hold one {name}", f"one {name} per distinct {key}"]
    if entity.null_key_rows:
        clauses.append(f"{entity.null_key_rows} rows have a NULL in {key}")
    return "; ".join([*clauses, *_ordering_clauses(name, identity)])


def _ordering_clauses(name: str, identity: EntityIdentity | None) -> list[str]:
    ordering = dict.fromkeys(
        a.by for a in (identity.attributes if identity else ())
        if a.rule == ResolutionRule.MAX_BY and a.by
    )
    return [
        f"each {name} has exactly one row per {by}, so the greatest {by} picks one row of each"
        for by in ordering
    ]


def _format_value(value: dict[str, str | None]) -> str:
    """The code to filter on, with its label, or `no label` when none says more than the code."""
    code, label = value["code"], value.get("label")
    return f"'{code}' ({label})" if label and label != code else f"'{code}' (no label)"


def _format_values_line(
    values: list[dict[str, str | None]], max_values: int | None,
) -> str:
    truncated = max_values is not None and len(values) > max_values
    if truncated:
        values = values[:max_values]
    values_str = ", ".join(_format_value(v) for v in values)
    suffix = " [truncated]" if truncated else ""
    return f"[{values_str}]{suffix}"


def _add_governed_values_section(
    parts: list[str],
    sco: SemanticContextObject,
    max_values: int | None = None,
) -> None:
    if sco.governed_values:
        parts.append(
            "\nCoded values show each code's label in parentheses: "
            "filter on the code; answers show the label."
        )
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


def _add_result_shape(
    parts: list[str], sco: SemanticContextObject,
) -> None:
    if sco.result_shape is None or not sco.result_shape.columns:
        return
    parts.append(
        "\nRESULT SHAPE (return exactly these columns, in this order):"
    )
    parts.extend(
        f"  {n}. {column}"
        for n, column in enumerate(sco.result_shape.columns, start=1)
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
    elif dialect == "duckdb":
        parts.append("\nDIALECT NOTES (DuckDB SQL):")
        parts.append("- Use the table names as listed: catalog.schema.table")
        parts.append(
            "- Use double quotes for an identifier that is a reserved word "
            "or holds a character other than letters, digits or underscores"
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
