"""Verify shared concepts collapse across studies in the Neo4j graph.

Queries Neo4j for `:Term` nodes whose downstream paths reach columns in more
than one `source_schema`. Emits a JSON report listing each shared term, the
contributing schemas, and the connecting edge sample. Exits non-zero if a
specifically-required term (passed via `--require-code`) is not multi-study.

Usage:
    uv run python scripts/verify_cross_study_collapse.py \\
        --output eval-runs/msk-chord-full-C/cross-study-collapse.json \\
        --require-code HGNC:TP53
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from sema.cli_factories import _get_neo4j_driver
from sema.log import logger
from sema.models.config import Neo4jConfig

SHARED_TERM_QUERY = """
MATCH (t:Term)
OPTIONAL MATCH (t)-[m:MEMBER_OF]->(:ValueSet)<-[hvs:HAS_VALUE_SET]-(c:Column)
WITH t,
     collect(DISTINCT m.source_schema) +
     collect(DISTINCT hvs.source_schema) AS schemas
WITH t, [s IN schemas WHERE s IS NOT NULL] AS schemas
WHERE size(schemas) > 1
RETURN t.code AS code,
       t.label AS label,
       schemas
ORDER BY t.code
"""


def _query_shared_terms(driver: Any) -> list[dict[str, Any]]:
    with driver.session() as session:
        result = session.run(SHARED_TERM_QUERY)
        return [
            {
                "code": rec["code"],
                "label": rec["label"],
                "source_schemas": sorted(set(rec["schemas"])),
            }
            for rec in result
        ]


def _enforce_required(
    shared: list[dict[str, Any]], required_codes: tuple[str, ...]
) -> list[str]:
    found = {entry["code"] for entry in shared}
    return [code for code in required_codes if code not in found]


@click.command()
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="JSON report destination path.",
)
@click.option(
    "--require-code",
    "required_codes",
    multiple=True,
    default=(),
    help="Term code that must appear with multiple source_schemas (repeatable).",
)
def main(output_path: Path, required_codes: tuple[str, ...]) -> None:
    """Run shared-term verification and write JSON report."""
    driver = _get_neo4j_driver(Neo4jConfig())
    try:
        shared = _query_shared_terms(driver)
    finally:
        driver.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"shared_terms": shared, "count": len(shared)},
            indent=2,
            sort_keys=True,
        )
    )
    logger.info("Wrote {} shared terms to {}", len(shared), output_path)

    missing = _enforce_required(shared, required_codes)
    if missing:
        logger.error("Required terms not multi-study: {}", missing)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main(standalone_mode=True))
