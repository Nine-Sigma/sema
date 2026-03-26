"""Export human override assertions for migration.

Queries assertions with status in (accepted, rejected, pinned)
and exports them as JSON for re-import after graph rebuild.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def export_overrides(
    driver: Any, output_path: str,
) -> list[dict[str, Any]]:
    """Export override assertions to JSON file.

    Returns the list of exported overrides.
    """
    overrides = _query_overrides(driver)
    Path(output_path).write_text(
        json.dumps(overrides, indent=2, default=str)
    )
    logger.info(
        "Exported %d override assertions to %s",
        len(overrides), output_path,
    )
    return overrides


def _query_overrides(
    driver: Any,
) -> list[dict[str, Any]]:
    with driver.session() as session:
        result = session.run(
            "MATCH (a:Assertion) "
            "WHERE a.status IN "
            "['accepted', 'rejected', 'pinned'] "
            "RETURN a.id AS id, "
            "a.subject_ref AS subject_ref, "
            "a.predicate AS predicate, "
            "a.payload AS payload, "
            "a.source AS source, "
            "a.confidence AS confidence, "
            "a.status AS status, "
            "a.run_id AS run_id, "
            "a.observed_at AS observed_at"
        )
        return [dict(record) for record in result]
