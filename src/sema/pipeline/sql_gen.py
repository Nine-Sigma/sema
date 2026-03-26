from __future__ import annotations

import json
import logging
from typing import Any

from sema.models.context import SemanticContextObject
from sema.pipeline.validate import validate_sql_against_sco

logger = logging.getLogger(__name__)


def build_sql_prompt(sco: SemanticContextObject, question: str) -> str:
    """Build constrained SQL generation prompt from an SCO."""
    parts = ["You are a SQL expert for Databricks. Generate a SQL query using "
             "ONLY the tables, columns, and values provided below. "
             "Do not use any table or column not listed.\n"]

    parts.append("AVAILABLE TABLES AND COLUMNS:")
    for asset in sco.physical_assets:
        fqn = f"{asset.catalog}.{asset.schema}.{asset.table}"
        parts.append(f"  {fqn}: {', '.join(asset.columns)}")

    if sco.join_paths:
        parts.append("\nJOIN PATHS:")
        for jp in sco.join_paths:
            parts.append(
                f"  {jp.from_table} -> {jp.to_table} "
                f"ON {jp.on_column} ({jp.cardinality})"
            )

    if sco.governed_values:
        parts.append("\nGOVERNED FILTER VALUES (use these exact values):")
        for gv in sco.governed_values:
            values_str = ", ".join(
                f"'{v['code']}'" for v in gv.values
            )
            parts.append(f"  {gv.table}.{gv.column}: [{values_str}]")

    parts.append("\nRULES:")
    parts.append("- Use ONLY the tables and columns listed above")
    parts.append("- Use ONLY the filter values provided — "
                 "do not guess or abbreviate")
    parts.append("- Join tables using the join paths provided")
    parts.append("- Column names are exact — "
                 "copy them precisely as shown")
    parts.append("- Use fully qualified table names "
                 "(catalog.schema.table)")

    parts.append(f"\nQuestion: {question}")
    parts.append("\nReturn ONLY the SQL query, no markdown or explanation.")

    return "\n".join(parts)


class SQLGenerator:
    """Generate constrained SQL from an SCO using an LLM."""

    def __init__(self, llm: Any = None) -> None:
        self._llm = llm

    def generate(
        self,
        sco: SemanticContextObject,
        question: str,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Generate and validate SQL with retry loop."""
        prompt = build_sql_prompt(sco, question)
        errors: list[str] = []

        for attempt in range(max_retries + 1):
            if attempt > 0:
                feedback = (
                    f"\n\nPrevious SQL had errors:\n"
                    + "\n".join(f"- {e}" for e in errors)
                    + "\n\nPlease fix and try again."
                )
                current_prompt = prompt + feedback
            else:
                current_prompt = prompt

            try:
                response = self._llm.invoke(current_prompt)
                sql = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                ).strip()

                if sql.startswith("```"):
                    lines = sql.split("\n")
                    sql = "\n".join(lines[1:-1]).strip()

                errors = validate_sql_against_sco(sql, sco)
                if not errors:
                    return {
                        "sql": sql,
                        "valid": True,
                        "errors": [],
                        "attempts": attempt + 1,
                    }

            except Exception as e:
                errors = [f"LLM error: {e}"]

        return {
            "sql": sql if "sql" in dir() else "",
            "valid": False,
            "errors": errors,
            "attempts": max_retries + 1,
        }
