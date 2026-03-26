from __future__ import annotations

import logging

import sqlglot

from sema.models.context import SemanticContextObject
from sema.pipeline.validate_utils import (
    _build_allowed_sets,
    _check_references,
)

logger = logging.getLogger(__name__)


def validate_sql_against_sco(
    sql: str, sco: SemanticContextObject
) -> list[str]:
    """Validate SQL references against SCO-provided assets.

    Returns list of error strings. Empty list = valid.
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect="databricks")
    except Exception as e:
        return [f"SQL syntax error: {e}"]

    allowed_tables, allowed_columns = _build_allowed_sets(sco)
    return _check_references(parsed, allowed_tables, allowed_columns)  # type: ignore[arg-type]
