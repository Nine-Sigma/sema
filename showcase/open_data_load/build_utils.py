"""The build command's report: stored joins, their evidence recomputed on the file, candidates.

The re-check reads the DuckDB file directly with one read-only connection; it shares no
code with the detector, so it can disagree with it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
from neo4j import GraphDatabase

from sema.connectors.duckdb_utils import (
    DUCKDB_PLATFORM,
    duckdb_workspace,
    qualified_table,
    quote_ident,
)
from sema.engine.join_detector import COVERAGE_THRESHOLD, DEFAULT_SAMPLE_CAP
from sema.engine.join_detector_utils import enumerate_candidates_from_metadata
from sema.graph.loader import GraphLoader
from sema.graph.loader_utils import fetch_columns_by_schema
from sema.models.config import BuildConfig
from sema.models.physical_key import CanonicalRef

JOINS_QUERY = (
    "MATCH (jp:JoinPath {source_schema: $schema})-[:USES]->(scoped:Table) "
    "WHERE scoped.ref STARTS WITH $prefix "
    "WITH DISTINCT jp "
    "OPTIONAL MATCH (jp)-[:USES]->(t:Table) "
    "WITH jp, collect(DISTINCT t.name) AS uses_tables, collect(DISTINCT t.ref) AS table_refs "
    "OPTIONAL MATCH (jp)-[:USES]->(c:Column) "
    "WITH jp, uses_tables, table_refs, collect(DISTINCT c.ref) AS column_refs "
    "OPTIONAL MATCH (jp)-[:FROM_ENTITY]->(fe:Entity) "
    "WITH jp, uses_tables, table_refs, column_refs, collect(DISTINCT fe.name) AS from_entities "
    "OPTIONAL MATCH (jp)-[:TO_ENTITY]->(te:Entity) "
    "RETURN jp.name AS name, jp.join_predicates AS join_predicates, jp.source AS source, "
    "jp.confidence AS confidence, uses_tables, table_refs, column_refs, from_entities, "
    "collect(DISTINCT te.name) AS to_entities ORDER BY name"
)
RECHECK_QUERY = (
    "WITH referenced AS (SELECT CAST({pk} AS VARCHAR) AS v FROM {pk_table}), "
    "referencing AS (SELECT DISTINCT CAST({fk} AS VARCHAR) AS v FROM {fk_table} "
    "WHERE {fk} IS NOT NULL) "
    "SELECT (SELECT count(DISTINCT v) FROM referenced), (SELECT count(v) FROM referenced), "
    "(SELECT count(*) FROM referenced), (SELECT count(*) FROM referencing), "
    "(SELECT count(*) FROM referencing WHERE v IN (SELECT v FROM referenced WHERE v IS NOT NULL))"
)


def table_prefix(database: Path, catalog: str, schema: str) -> str:
    datasource = CanonicalRef.datasource(DUCKDB_PLATFORM, duckdb_workspace(str(database)))
    return f"{datasource}/{catalog}/{schema}/"


def recheck(
    con: duckdb.DuckDBPyConnection, catalog: str, schema: str, predicate: dict[str, str],
) -> dict[str, Any]:
    """Full-column uniqueness of the referenced side and full containment of the other."""
    row = con.execute(RECHECK_QUERY.format(
        pk=quote_ident(predicate["referenced_column"]),
        pk_table=qualified_table(catalog, schema, predicate["referenced_table"]),
        fk=quote_ident(predicate["referencing_column"]),
        fk_table=qualified_table(catalog, schema, predicate["referencing_table"]),
    )).fetchone()
    assert row is not None
    distinct, non_null, rows, referencing, contained = (int(value) for value in row)
    return {
        "referenced": {"distinct_non_null": distinct, "non_null": non_null, "rows": rows},
        "referencing_distinct_non_null": referencing,
        "contained": contained,
        "containment_ratio": contained / referencing if referencing else None,
    }


def stored_predicates(raw: str | None) -> list[dict[str, str]]:
    """The materializer stores the referenced relation on the left."""
    return [
        {"referenced_table": p["left_table"], "referenced_column": p["left_column"],
         "referencing_table": p["right_table"], "referencing_column": p["right_column"]}
        for p in json.loads(raw or "[]")
    ]  # fmt: skip


def unparsed(refs: list[str]) -> list[str]:
    failures = []
    for ref in refs:
        try:
            CanonicalRef.parse(ref)
        except ValueError:
            failures.append(ref)
    return failures


def stored_join(
    row: dict[str, Any], con: duckdb.DuckDBPyConnection, catalog: str, schema: str,
) -> dict[str, Any]:
    refs = sorted(ref for ref in [*row["table_refs"], *row["column_refs"]] if ref is not None)
    failures = unparsed(refs)
    return {
        "name": row["name"], "source": row["source"], "confidence": row["confidence"],
        "predicates": [
            {**p, "recheck": recheck(con, catalog, schema, p)}
            for p in stored_predicates(row["join_predicates"])
        ],
        "uses_tables": sorted(name for name in row["uses_tables"] if name is not None),
        "from_entities": sorted(row["from_entities"]),
        "to_entities": sorted(row["to_entities"]),
        "refs": refs, "refs_parse": not failures, "unparsed_refs": failures,
    }


def candidates(
    loader: GraphLoader, con: duckdb.DuckDBPyConnection, catalog: str, schema: str,
    prefix: str,
) -> list[dict[str, Any]]:
    """What the detector proposes from the columns `run_fk_detection` reads for the schema."""
    found = []
    for c in enumerate_candidates_from_metadata(fetch_columns_by_schema(loader, schema)):
        predicate = {"referenced_table": c.pk_table, "referenced_column": c.pk_column,
                     "referencing_table": c.fk_table, "referencing_column": c.fk_column}
        # The read matches the schema name alone, so another file's same-named schema leaks in.
        here = c.pk_ref.startswith(prefix) and c.fk_ref.startswith(prefix)
        found.append({**predicate, "pk_ref": c.pk_ref, "fk_ref": c.fk_ref,
                      "recheck": recheck(con, catalog, schema, predicate) if here else None})
    return found


@contextmanager
def _graph(config: BuildConfig) -> Iterator[GraphLoader]:
    neo4j = config.neo4j
    driver = GraphDatabase.driver(neo4j.uri, auth=(neo4j.user, neo4j.password.get_secret_value()))
    try:
        yield GraphLoader(driver)
    finally:
        driver.close()


def join_report(config: BuildConfig, database: Path, build: dict[str, Any]) -> dict[str, Any]:
    schema = config.schemas[0]
    prefix = table_prefix(database, config.catalog, schema)
    with _graph(config) as loader, duckdb.connect(str(database), read_only=True) as con:
        rows = loader._run_read(JOINS_QUERY, schema=schema, prefix=prefix)
        joins = [stored_join(row, con, config.catalog, schema) for row in rows]
        proposed = candidates(loader, con, config.catalog, schema, prefix)
    return {
        "datasource_tables": prefix, "build": build,
        "thresholds": {"coverage": COVERAGE_THRESHOLD, "sample_cap": DEFAULT_SAMPLE_CAP,
                       "materialization": config.fk_materialization_threshold},
        "join_count": len(joins), "joins": joins,
        "candidate_count": len(proposed), "candidates": proposed,
    }
