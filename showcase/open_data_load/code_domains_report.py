"""Report every binding SEMA proposes for a coded column, whether the build graph links it, and
how much of the column's data its value set lists.

Run as `python -m showcase.open_data_load.code_domains_report --profile <yaml> --spec <yaml>
--out <md>`. The profile's `build:` keys name the DuckDB file, catalog and schema; the spec
is intake's spec for the source (a `concept_source` file); Neo4j comes from the `NEO4J_*`
environment. Proposals come from the spec's stated domains, the declared keys the connector
reads, and the build graph's own value-set links. The command builds nothing.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from neo4j import GraphDatabase, RoutingControl

from sema.binding.spec import ConceptSourceSpec, load_concept_source_spec
from sema.connectors.duckdb import DuckDBConnector
from sema.models.config import DuckDBConfig, Neo4jConfig
from showcase.open_data_load.build import BuildTarget, read_build_target
from showcase.open_data_load.build_utils import table_prefix
from showcase.open_data_load.code_domains_report_utils import (
    BUILD_GRAPH,
    Proposal,
    WarehouseTable,
    code_values,
    coverage,
    declared_keys,
    merged,
    stated_domains,
)

LINKS_QUERY = (
    "MATCH (c:Column)-[:HAS_VALUE_SET]->(vs:ValueSet) WHERE c.ref STARTS WITH $prefix "
    "OPTIONAL MATCH (t:Term)-[:MEMBER_OF]->(vs) "
    "RETURN c.ref AS ref, vs.name AS value_set, collect(DISTINCT t.code) AS codes, "
    "count(DISTINCT CASE WHEN t.label IS NOT NULL THEN t END) AS labeled"
)
COLUMNS = (
    "column", "proposed value set", "evidence", "linked", "labels",
    "listed_value_share", "listed_row_share", "null_rows", "reason",
)


@dataclass(frozen=True)
class GraphLink:
    ref: str
    value_set: str
    codes: frozenset[str]
    labeled: int


@dataclass(frozen=True)
class Row:
    column: str | None
    value_set: str
    evidence: tuple[str, ...]
    linked: bool
    labels: bool
    listed_value_share: float | None
    listed_row_share: float | None
    null_rows: int | None
    reason: str


def _connector(target: BuildTarget) -> DuckDBConnector:
    return DuckDBConnector(DuckDBConfig(path=str(target.database)))


def _tables(connector: DuckDBConnector, target: BuildTarget) -> dict[str, WarehouseTable]:
    found = {}
    for item in connector.discover_tables(target.catalog, [target.schema]):
        columns = connector.extract_columns_dto(target.catalog, target.schema, item.table_name)
        found[item.table_name] = WarehouseTable(
            item.table_name, item.fqn, tuple(c.name for c in columns)
        )
    return found


def warehouse_tables(target: BuildTarget) -> list[WarehouseTable]:
    connector = _connector(target)
    try:
        return list(_tables(connector, target).values())
    finally:
        connector.close()


def read_links(neo4j: Neo4jConfig, prefix: str) -> list[GraphLink]:
    auth = (neo4j.user, neo4j.password.get_secret_value())
    with GraphDatabase.driver(neo4j.uri, auth=auth) as driver:
        records = driver.execute_query(
            LINKS_QUERY, prefix=prefix, routing_=RoutingControl.READ
        ).records
    return [
        GraphLink(r["ref"], r["value_set"], frozenset(r["codes"]), r["labeled"]) for r in records
    ]


def _column_refs(tables: dict[str, WarehouseTable]) -> dict[tuple[str, str], str]:
    return {(t.name, c): f"{t.ref}/{c}" for t in tables.values() for c in t.columns}


def _with_links(
    proposed: list[Proposal], links: list[GraphLink], tables: dict[str, WarehouseTable],
) -> list[tuple[Proposal, GraphLink | None]]:
    """Each proposal with the graph link on its column; a link nothing proposed is its own."""
    refs = _column_refs(tables)
    link_of = {link.ref: link for link in links}
    pairs: list[tuple[Proposal, GraphLink | None]] = []
    placed = set()
    for p in proposed:
        ref = refs.get((p.table or "", p.column or ""))
        placed.add(ref)
        pairs.append((p, link_of.get(ref) if ref else None))
    column_of = {ref: key for key, ref in refs.items()}
    for link in links:
        if link.ref in placed:
            continue
        table, column = column_of.get(link.ref, (None, None))
        problems = () if column else (f"linked column {link.ref} is not in the warehouse",)
        pairs.append((
            Proposal(table, column, link.value_set, None, frozenset({BUILD_GRAPH}), problems),
            link,
        ))  # fmt: skip
    return pairs


def _link_reasons(link: GraphLink | None) -> list[str]:
    if link is None:
        return ["no HAS_VALUE_SET"]
    if not link.codes:
        return [f"value set {link.value_set} has no members"]
    if link.labeled < len(link.codes):
        return [f"{link.labeled} of {len(link.codes)} members of {link.value_set} carry a label"]
    return []


def _row(
    proposal: Proposal, link: GraphLink | None, connector: DuckDBConnector, target: BuildTarget,
) -> Row:
    linked = link is not None and bool(link.codes)
    codes = link.codes if link is not None and link.codes else None
    if codes is None and proposal.codes_at is not None:
        codes = frozenset(code_values(connector, target.catalog, target.schema, proposal.codes_at))
    numbers = None
    if proposal.table and proposal.column:
        numbers = coverage(
            connector, target.catalog, target.schema, proposal.table, proposal.column, codes
        )
    return Row(
        column=f"{proposal.table}.{proposal.column}" if proposal.column else None,
        value_set=proposal.value_set,
        evidence=tuple(sorted(proposal.evidence)),
        linked=linked,
        labels=linked and link is not None and link.labeled == len(link.codes),
        listed_value_share=numbers.listed_value_share if numbers else None,
        listed_row_share=numbers.listed_row_share if numbers else None,
        null_rows=numbers.null_rows if numbers else None,
        reason="; ".join([*proposal.problems, *_link_reasons(link)]),
    )


def report(target: BuildTarget, spec: ConceptSourceSpec, links: list[GraphLink]) -> list[Row]:
    connector = _connector(target)
    try:
        tables = _tables(connector, target)
        proposed = merged([
            *stated_domains(spec, tables),
            *declared_keys(connector, target.catalog, target.schema, tables),
        ])  # fmt: skip
        pairs = _with_links(proposed, links, tables)
        rows = [_row(proposal, link, connector, target) for proposal, link in pairs]
    finally:
        connector.close()
    return sorted(rows, key=lambda r: (r.column or "", r.value_set, r.reason))


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, tuple):
        return ", ".join(value)
    return str(value).replace("|", "\\|")


def render(rows: list[Row], heading: list[str]) -> str:
    by_evidence = Counter(kind for row in rows for kind in row.evidence)
    summary = [
        f"- proposed bindings: {len(rows)}",
        *(f"- proposed by {kind}: {n}" for kind, n in sorted(by_evidence.items())),
        f"- linked: {sum(row.linked for row in rows)}",
        f"- linked with labels: {sum(row.labels for row in rows)}",
    ]
    table = [
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
        *(
            "| " + " | ".join(_cell(v) for v in (
                row.column, row.value_set, row.evidence, row.linked, row.labels,
                row.listed_value_share, row.listed_row_share, row.null_rows, row.reason,
            )) + " |"
            for row in rows
        ),
    ]  # fmt: skip
    return "\n".join(["# Code domains report", "", *heading, "", *summary, "", *table, ""])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    target = read_build_target(args.profile)
    prefix = table_prefix(target.database, target.catalog, target.schema)
    rows = report(target, load_concept_source_spec(args.spec), read_links(Neo4jConfig(), prefix))
    heading = [
        f"- profile: `{args.profile}`", f"- spec: `{args.spec}`",
        f"- warehouse: `{target.database}`, catalog `{target.catalog}`, schema `{target.schema}`",
        f"- graph columns under: `{prefix}`",
    ]  # fmt: skip
    args.out.write_text(render(rows, heading), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
