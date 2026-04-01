"""Result synthesis — natural language summary of query results."""

from __future__ import annotations

import json
from typing import Any

from sema.log import logger


def synthesize_results(
    question: str,
    sql: str,
    results: dict[str, Any],
    llm: Any,
) -> str:
    """Summarize query results in natural language."""
    row_count = results.get("row_count", 0)
    prompt = _build_synthesis_prompt(question, sql, results, row_count)

    try:
        response = llm.invoke(prompt)
        return (
            response.content
            if hasattr(response, "content")
            else str(response)
        )
    except Exception as e:
        logger.warning("Result synthesis failed: {}", e)
        return f"Query returned {row_count} rows."


def _build_synthesis_prompt(
    question: str,
    sql: str,
    results: dict[str, Any],
    row_count: int,
) -> str:
    if row_count == 0:
        return (
            f"The user asked: {question}\n\n"
            f"The SQL query was:\n{sql}\n\n"
            f"The query returned zero rows. "
            f"Explain what was searched for and that no matching "
            f"records were found. Suggest possible reasons."
        )
    sample = results.get("rows", [])[:10]
    return (
        f"The user asked: {question}\n\n"
        f"The SQL query returned {row_count} rows. "
        f"Here are the first rows:\n"
        f"{json.dumps(sample, indent=2, default=str)}\n\n"
        f"Summarize the results in the context of the "
        f"original question. Be concise."
    )
