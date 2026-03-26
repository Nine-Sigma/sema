"""Import human override assertions after migration.

Translates old-format refs to new format, matches by dedupe key
(subject_ref, predicate, source), restores status, and logs
orphans.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sema.models.constants import translate_ref

logger = logging.getLogger(__name__)


def import_overrides(
    driver: Any,
    input_path: str,
    workspace: str,
    dry_run: bool = False,
) -> dict[str, int]:
    """Import overrides from JSON, translate refs, restore status.

    Returns counts: {"restored": N, "orphaned": M}.
    """
    overrides = json.loads(Path(input_path).read_text())
    restored = 0
    orphaned = 0

    for override in overrides:
        old_ref = override["subject_ref"]
        new_ref = translate_ref(old_ref, workspace)
        predicate = override["predicate"]
        source = override["source"]
        status = override["status"]

        match = _find_matching_assertion(
            driver, new_ref, predicate, source,
        )
        if match:
            if not dry_run:
                _restore_status(driver, match["id"], status)
            restored += 1
            logger.info(
                "Restored %s on (%s, %s, %s)",
                status, new_ref, predicate, source,
            )
        else:
            orphaned += 1
            logger.warning(
                "Orphaned override: old_ref=%s new_ref=%s "
                "predicate=%s source=%s status=%s",
                old_ref, new_ref, predicate, source, status,
            )

    logger.info(
        "Import complete: %d restored, %d orphaned",
        restored, orphaned,
    )
    return {"restored": restored, "orphaned": orphaned}


def _find_matching_assertion(
    driver: Any, subject_ref: str,
    predicate: str, source: str,
) -> dict[str, Any] | None:
    with driver.session() as session:
        result = session.run(
            "MATCH (a:Assertion) "
            "WHERE a.subject_ref = $ref "
            "AND a.predicate = $predicate "
            "AND a.source = $source "
            "AND a.status = 'auto' "
            "RETURN a.id AS id "
            "LIMIT 1",
            ref=subject_ref,
            predicate=predicate,
            source=source,
        )
        record = result.single()
        return dict(record) if record else None


def _restore_status(
    driver: Any, assertion_id: str, status: str,
) -> None:
    with driver.session() as session:
        session.run(
            "MATCH (a:Assertion {id: $id}) "
            "SET a.status = $status",
            id=assertion_id, status=status,
        )
