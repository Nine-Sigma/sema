"""Report every binding the build proposes for a coded column, whether the build graph links
it, with its verification and how much of the column's data its value set lists.

Run as `python -m showcase.open_data_load.code_domains_report --profile <yaml> --out <md>`.
The profile's `build:` keys name the catalog and schema; Neo4j comes from the `NEO4J_*`
environment. Everything comes from the graph the build wrote: the `has_value_set` assertions
(declared keys, stated domains, name matches, containment, unplaced domains) and the
`HAS_VALUE_SET` links with their `MEMBER_OF` members. The command builds nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase, RoutingControl

from sema.models.config import Neo4jConfig
from showcase.open_data_load.build import read_build_target
from showcase.open_data_load.code_domains_report_utils import (
    Evidence,
    GraphLink,
    Member,
    Row,
    binding_row,
    bindings,
    link_row,
)

__all__ = ["Evidence", "GraphLink", "Member", "Row", "main", "read_evidence", "read_links",
           "render", "report"]

EVIDENCE_QUERY = (
    "MATCH (a:Assertion {predicate: 'has_value_set', source_schema: $schema}) "
    "OPTIONAL MATCH (c:Column {ref: a.subject_ref}) "
    "RETURN c.table_name AS table, c.name AS column, a.object_ref AS value_set, "
    "a.payload AS payload"
)
LINKS_QUERY = (
    "MATCH (c:Column {schema_name: $schema, catalog: $catalog})-[h:HAS_VALUE_SET]->"
    "(vs:ValueSet) "
    "OPTIONAL MATCH (t:Term)-[m:MEMBER_OF]->(vs) "
    "RETURN c.table_name AS table, c.name AS column, vs.name AS value_set, "
    "properties(h) AS props, "
    "collect(CASE WHEN t IS NULL THEN NULL ELSE [t.code, t.label, m.rows, m.listed] END) "
    "AS members"
)
COLUMNS = (
    "column", "proposed value set", "evidence", "linked", "verification", "labels",
    "listed_value_share", "listed_row_share", "null_rows", "reason",
)


def _records(neo4j: Neo4jConfig, query: str, **params: Any) -> list[Any]:
    auth = (neo4j.user, neo4j.password.get_secret_value())
    with GraphDatabase.driver(neo4j.uri, auth=auth) as driver:
        return list(driver.execute_query(query, routing_=RoutingControl.READ, **params).records)


def read_evidence(neo4j: Neo4jConfig, schema: str) -> list[Evidence]:
    found = []
    for r in _records(neo4j, EVIDENCE_QUERY, schema=schema):
        payload = json.loads(r["payload"]) if isinstance(r["payload"], str) else r["payload"]
        column = f"{r['table']}.{r['column']}" if r["column"] else None
        found.append(Evidence(column, r["value_set"], payload.get("evidence", ""), payload))
    return found


def read_links(neo4j: Neo4jConfig, catalog: str, schema: str) -> list[GraphLink]:
    return [
        GraphLink(
            f"{r['table']}.{r['column']}", r["value_set"], dict(r["props"]),
            tuple(Member(str(code), label, rows, listed)
                  for code, label, rows, listed in r["members"]),
        )
        for r in _records(neo4j, LINKS_QUERY, catalog=catalog, schema=schema)
    ]


def report(evidence: list[Evidence], links: list[GraphLink]) -> list[Row]:
    proposed = bindings(evidence)
    linked = {b.column: b for b in proposed if b.linked and b.column is not None}
    linked_columns = {link.column for link in links}
    rows = [link_row(link, linked.get(link.column)) for link in links]
    rows += [
        binding_row(b) for b in proposed
        if not (b.linked and b.column in linked_columns)
    ]
    return sorted(rows, key=lambda r: (r.column or "", r.value_set, r.reason))


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, tuple):
        return ", ".join(value)
    return str(value).replace("|", "\\|")


def render(rows: list[Row], heading: list[str]) -> str:
    by_evidence = Counter(kind for row in rows for kind in row.evidence)
    verified = sum(row.linked and row.verification == "verified" for row in rows)
    summary = [
        f"- proposed bindings: {len(rows)}",
        *(f"- proposed by {kind}: {n}" for kind, n in sorted(by_evidence.items())),
        f"- linked: {sum(row.linked for row in rows)}",
        f"- linked and verified: {verified}",
        f"- linked with labels: {sum(row.labels for row in rows)}",
    ]
    table = [
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
        *(
            "| " + " | ".join(_cell(v) for v in (
                row.column, row.value_set, row.evidence, row.linked, row.verification,
                row.labels, row.listed_value_share, row.listed_row_share, row.null_rows,
                row.reason,
            )) + " |"
            for row in rows
        ),
    ]  # fmt: skip
    return "\n".join(["# Code domains report", "", *heading, "", *summary, "", *table, ""])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    target = read_build_target(args.profile)
    neo4j = Neo4jConfig()
    rows = report(
        read_evidence(neo4j, target.schema), read_links(neo4j, target.catalog, target.schema),
    )
    heading = [
        f"- profile: `{args.profile}`",
        f"- graph: catalog `{target.catalog}`, schema `{target.schema}`",
    ]
    args.out.write_text(render(rows, heading), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
