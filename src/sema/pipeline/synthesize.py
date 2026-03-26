from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def synthesize_results(
    llm: Any,
    question: str,
    sql: str,
    results: dict[str, Any],
) -> str:
    """Summarize query results in natural language."""
    row_count = results.get("row_count", 0)

    if row_count == 0:
        prompt = (
            f"The user asked: {question}\n\n"
            f"The SQL query was:\n{sql}\n\n"
            f"The query returned zero rows. "
            f"Explain what was searched for and that no matching "
            f"records were found. Suggest possible reasons."
        )
    else:
        sample = results.get("rows", [])[:10]
        prompt = (
            f"The user asked: {question}\n\n"
            f"The SQL query returned {row_count} rows. "
            f"Here are the first rows:\n"
            f"{json.dumps(sample, indent=2, default=str)}\n\n"
            f"Summarize the results in the context of the "
            f"original question. Be concise."
        )

    try:
        response = llm.invoke(prompt)
        return (
            response.content
            if hasattr(response, "content")
            else str(response)
        )
    except Exception as e:
        logger.warning(f"Result synthesis failed: {e}")
        return f"Query returned {row_count} rows."
