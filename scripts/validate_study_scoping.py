"""Fail if any study-derived graph state still has source_schema = NULL.

Run AFTER migrations/002_backfill_null_scoped_study_edges.cypher. A null
``source_schema`` on study-derived relationships/nodes is invisible to
``delete_study_scoped``, so it can never be removed by a study rebuild. This
script makes that check executable and BLOCKING (exit 1) rather than a comment.

Gating principle: gate every relationship that touches a STUDY-ONLY node
(:Column / :Table / :Entity / :Property / :ValueSet / :Alias) plus :JoinPath
nodes — these exist only because of study materialization, so a null
source_schema is always study residue. Gated set: ENTITY_ON_TABLE,
HAS_PROPERTY, PROPERTY_ON_COLUMN, HAS_VALUE_SET, MEMBER_OF, CLASSIFIED_AS
(Property-anchored), REFERS_TO into a study Entity/Property, and :JoinPath.

Deliberately NOT gated: IN_VOCABULARY and PARENT_OF (both endpoints are
:Term / :Vocabulary, which ontology preload legitimately leaves null-scoped)
and the Alias->Term subcase of REFERS_TO. These are not locally
distinguishable from study residue and are surfaced by the informational
diagnostics in migrations/002 for manual review.

Usage:
    uv run python scripts/validate_study_scoping.py
"""
from __future__ import annotations

import sys
from typing import Any

import click

from sema.cli_factories import _get_neo4j_driver
from sema.log import logger
from sema.models.config import Neo4jConfig

# Study-derived shapes that MUST be scoped. Each entry: label -> Cypher that
# returns rows describing remaining null-scoped instances.
_GATED_QUERIES: dict[str, str] = {
    "MEMBER_OF": (
        "MATCH (:Term)-[m:MEMBER_OF]->(vs:ValueSet) "
        "WHERE m.source_schema IS NULL "
        "RETURN vs.column_ref AS value_set_ref, vs.name AS value_set_name, "
        "count(*) AS n"
    ),
    "HAS_VALUE_SET": (
        "MATCH (:Column)-[r:HAS_VALUE_SET]->(:ValueSet) "
        "WHERE r.source_schema IS NULL RETURN count(r) AS n"
    ),
    "PROPERTY_ON_COLUMN": (
        "MATCH (:Property)-[r:PROPERTY_ON_COLUMN]->(:Column) "
        "WHERE r.source_schema IS NULL RETURN count(r) AS n"
    ),
    "HAS_PROPERTY": (
        "MATCH (:Entity)-[r:HAS_PROPERTY]->(:Property) "
        "WHERE r.source_schema IS NULL RETURN count(r) AS n"
    ),
    "ENTITY_ON_TABLE": (
        "MATCH (:Entity)-[r:ENTITY_ON_TABLE]->(:Table) "
        "WHERE r.source_schema IS NULL RETURN count(r) AS n"
    ),
    # Property is study-only (built from columns), so a Property's
    # CLASSIFIED_AS is always study-derived even though Vocabulary may be
    # preloaded.
    "CLASSIFIED_AS": (
        "MATCH (:Property)-[r:CLASSIFIED_AS]->(:Vocabulary) "
        "WHERE r.source_schema IS NULL RETURN count(r) AS n"
    ),
    # REFERS_TO into a study node (Entity/Property) is study-derived; the
    # Alias->Term subcase may be preload and is left informational.
    "REFERS_TO": (
        "MATCH (:Alias)-[r:REFERS_TO]->(target) "
        "WHERE r.source_schema IS NULL "
        "AND (target:Entity OR target:Property) RETURN count(r) AS n"
    ),
    "JoinPath": (
        "MATCH (jp:JoinPath) WHERE jp.source_schema IS NULL "
        "RETURN count(jp) AS n"
    ),
    # JoinPath's own edges are study-derived and source_schema-stamped.
    "USES": (
        "MATCH (:JoinPath)-[r:USES]->() WHERE r.source_schema IS NULL "
        "RETURN count(r) AS n"
    ),
    "FROM_ENTITY": (
        "MATCH (:JoinPath)-[r:FROM_ENTITY]->() WHERE r.source_schema IS NULL "
        "RETURN count(r) AS n"
    ),
    "TO_ENTITY": (
        "MATCH (:JoinPath)-[r:TO_ENTITY]->() WHERE r.source_schema IS NULL "
        "RETURN count(r) AS n"
    ),
}


def _query_residual(driver: Any) -> dict[str, list[dict[str, Any]]]:
    residual: dict[str, list[dict[str, Any]]] = {}
    with driver.session() as session:
        for label, cypher in _GATED_QUERIES.items():
            rows = [dict(r) for r in session.run(cypher)]
            offending = [r for r in rows if r.get("n", 0)]
            if offending:
                residual[label] = offending
    return residual


def summarize_residual(residual: dict[str, list[dict[str, Any]]]) -> str:
    """Render residual findings; empty string means the graph is clean."""
    if not residual:
        return ""
    lines = ["Null-scoped study state remains (cannot be study-deleted):"]
    for label, rows in residual.items():
        total = sum(int(r.get("n", 0)) for r in rows)
        lines.append(f"  {label}: {total}")
        for r in rows:
            detail = {k: v for k, v in r.items() if k != "n"}
            if detail:
                lines.append(f"    - {detail} (n={r.get('n')})")
    return "\n".join(lines)


@click.command()
def main() -> None:
    """Exit non-zero if study-derived null-scoped state remains."""
    driver = _get_neo4j_driver(Neo4jConfig())
    try:
        residual = _query_residual(driver)
    finally:
        driver.close()

    report = summarize_residual(residual)
    if report:
        logger.error("{}", report)
        sys.exit(1)
    logger.info("Study scoping clean: no null-scoped study state remains.")


if __name__ == "__main__":
    sys.exit(main(standalone_mode=True))
