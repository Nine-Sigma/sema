"""Reads behind the discovery scorecard: graph joins, declared keys, data rechecks, oracle files.

Keys follow the oracle file format: relation and column names lowercased, a column list as its
sorted names joined by `+`. Every warehouse read is a Databricks-dialect statement run through
`Connector.execute_query`, which each connector translates to its own dialect.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from sema.connectors.base import Connector
from sema.engine.coded_column_utils import CONTAINMENT, label_reading
from sema.engine.identity_utils import Counts
from sema.engine.warehouse_lookup import (
    ColumnKey,
    RelationEvidence,
    WarehouseProfileLookup,
    WarehouseValues,
)
from sema.intake.run.scorecard import Cells, Key
from sema.log import logger
from sema.models.extraction import ExtractedForeignKey
from sema.models.physical_key import CanonicalRef, PhysicalKey

COLUMN_SEPARATOR = "+"
# One row per has_value_set assertion on a column of the scope; a nominating evidence's row is
# dropped by `binding_read`. Members are read from the verified value set only.
CODED_BINDINGS = (
    "MATCH (a:Assertion {predicate: $predicate}) "
    "WHERE a.status <> 'superseded' AND ($schemas IS NULL OR a.source_schema IN $schemas) "
    "MATCH (c:Column {ref: a.subject_ref}) WHERE $catalog IS NULL OR c.catalog = $catalog "
    "OPTIONAL MATCH (c)-[:LABELED_BY {verification: $verified}]->(l:Column) "
    "RETURN a.payload AS payload, c.schema_name AS schema, c.table_name AS relation, "
    "c.name AS column, [l.schema_name, l.table_name, l.name] AS label, "
    "[(c)-[:HAS_VALUE_SET {verification: $verified}]->(:ValueSet)<-[m:MEMBER_OF]-(t:Term) "
    "| [t.code, t.vocabulary_name, t.label, m.rows]] AS members"
)


class BindingRead(NamedTuple):
    """A coded column's binding as the build graph holds it: the code source and verification
    its containment assertion states, its verified label column (schema, relation, column),
    whether its members carry its dictionary vocabulary's labels, and each member's rows."""

    schema: str
    relation: str
    column: str
    code_source: str
    verification: str
    label: tuple[str, str, str] | None
    dictionary_label: bool
    listed_row_share: float | None
    member_rows: dict[str, int]


def binding_read(record: Mapping[str, Any]) -> BindingRead | None:
    """A `CODED_BINDINGS` row as a binding; None for a nominating evidence's assertion."""
    payload = json.loads(record["payload"])
    if payload.get("evidence") != CONTAINMENT:
        return None
    source, members, label = str(payload["value_set"]), record["members"], record["label"]
    return BindingRead(
        schema=record["schema"], relation=record["relation"], column=record["column"],
        code_source=source, verification=str(payload.get("verification")),
        label=(label[0], label[1], label[2]) if label[0] is not None else None,
        dictionary_label=any(vocabulary == source and text is not None
                             for _, vocabulary, text, _ in members),
        listed_row_share=payload.get("listed_row_share"),
        member_rows={str(code): rows for code, _, _, rows in members if rows is not None},
    )


def code_relation(code_source: str) -> PhysicalKey | None:
    """The key column a code source refers to, or None for a dictionary vocabulary."""
    try:
        return CanonicalRef.parse(code_source)
    except ValueError:
        return None


def listed_codes(
    values: WarehouseValues, code_lists: Mapping[str, Mapping[str, Any]], code_source: str,
) -> set[str] | None:
    """A code source's codes: its key column's values, or its dictionary vocabulary's codes."""
    key = code_relation(code_source)
    if key is None:
        return set(code_lists.get(code_source, {}))
    counts = values.counts((key.schema or "", key.table, key.column or ""))
    return None if counts is None else {code for code in counts if code is not None}


def label_holds(values: WarehouseValues, read: BindingRead) -> bool:
    """Whether the code determines the label column: beside it, or beside the code source's
    key column in the code relation."""
    if read.label is None:
        return False
    schema, relation, column = read.label
    code = read.column
    if (schema, relation) != (read.schema, read.relation):
        key = code_relation(read.code_source)
        if key is None or (key.schema, key.table) != (schema, relation):
            return False
        code = key.column or ""
    return label_reading(values.pairs(schema, relation, code, column)) is not None


def columns_by_row_count(
    connector: Connector, catalog: str, schemas: list[str] | None, sizes: set[int],
) -> dict[int, list[ColumnKey]]:
    """Every column of the scope's relations whose row count is one of `sizes`, by that count."""
    profile = WarehouseProfileLookup(connector.execute_query, catalog)
    found: dict[int, list[ColumnKey]] = {}
    for table in connector.discover_tables(catalog, schemas or None):
        columns = [(table.schema, table.table_name, column.name) for column in
                   connector.extract_columns_dto(catalog, table.schema, table.table_name)]
        measured = profile(columns[0]) if columns else None
        if measured is not None and measured[1] in sizes:
            found.setdefault(measured[1], []).extend(columns)
    return found


# Only the build's join writer (`batch_upsert_join_paths`) sets `status`; the target loader's
# does not. Map access, so a graph where no join has a cardinality raises no unknown-key warning.
GRAPH_JOINS = (
    "MATCH (jp:JoinPath {status: 'ACTIVE'}) "
    "WHERE ($schemas IS NULL OR jp.source_schema IN $schemas) "
    "AND ($catalog IS NULL OR EXISTS { "
    "MATCH (jp)-[:USES]->(:Table {catalog: $catalog, schema_name: jp.source_schema}) }) "
    "RETURN jp.join_predicates AS predicates, properties(jp).cardinality_hint AS cardinality, "
    "jp.source AS join_source"
)


def columns_key(columns: Iterable[str]) -> str:
    return COLUMN_SEPARATOR.join(sorted(column.lower() for column in columns))


def join_key(
    relation: str, columns: Iterable[str], referenced: str, referenced_columns: Iterable[str],
) -> Key:
    return (relation.lower(), columns_key(columns), referenced.lower(),
            columns_key(referenced_columns))


def graph_join_key(predicates_json: str) -> Key:
    """A JoinPath's key; its predicates hold the referenced side left, the referencing right."""
    predicates: list[dict[str, Any]] = json.loads(predicates_json)
    return join_key(
        predicates[0]["right_table"], [p["right_column"] for p in predicates],
        predicates[0]["left_table"], [p["left_column"] for p in predicates],
    )


def declared_join_key(key: ExtractedForeignKey) -> Key:
    referencing, referenced = zip(*key.column_pairs, strict=True)
    return join_key(key.referencing_table, referencing, key.referenced_table, referenced)


def duplicate_count_query(catalog: str, column: tuple[str, str, str]) -> str:
    """Non-null values of one (schema, table, column) beyond the first of each value."""
    schema, table, name = column
    return (
        f"SELECT COUNT(`{name}`) - COUNT(DISTINCT `{name}`) "
        f"FROM `{catalog}`.`{schema}`.`{table}`"
    )


def orphan_count_query(
    catalog: str, referencing: tuple[str, str, str], referenced: tuple[str, str, str],
) -> str:
    """Non-null referencing values with no equal referenced value; each side (schema, table,
    column), compared as text so a key typed differently on each side still compares."""
    (schema, table, column), (ref_schema, ref_table, ref_column) = referencing, referenced
    return (
        f"SELECT COUNT(*) FROM `{catalog}`.`{schema}`.`{table}` r "
        f"WHERE r.`{column}` IS NOT NULL AND NOT EXISTS ("
        f"SELECT 1 FROM `{catalog}`.`{ref_schema}`.`{ref_table}` p "
        f"WHERE CAST(p.`{ref_column}` AS STRING) = CAST(r.`{column}` AS STRING))"
    )


def join_holds(
    connector: Connector, catalog: str, tables: Mapping[str, tuple[str, str]], key: Key,
    predicates_json: str | None = None,
) -> bool:
    """Whether a one-column join's referenced column is unique on the data and holds every
    non-NULL referencing value. A join on several columns is rechecked only from its
    predicates, which keep its column pairs."""
    relation, columns, referenced, referenced_columns = key
    if relation not in tables or referenced not in tables:
        return False
    if COLUMN_SEPARATOR in columns:
        return predicates_json is not None and _pairs_hold(
            Counts(RelationEvidence(connector, catalog)), tables[relation], tables[referenced],
            json.loads(predicates_json))
    referenced_column = (*tables[referenced], referenced_columns)
    queries = (
        duplicate_count_query(catalog, referenced_column),
        orphan_count_query(catalog, (*tables[relation], columns), referenced_column),
    )
    try:
        counts = [connector.execute_query(query) for query in queries]
    except Exception as exc:  # each connector raises its own driver's error type
        logger.warning("Join recheck on {} failed: {}", key, exc)
        return False
    return all(rows and int(rows[0][0]) == 0 for rows in counts)


def _pairs_hold(
    counts: Counts, child: tuple[str, str], parent: tuple[str, str],
    predicates: list[dict[str, Any]],
) -> bool:
    """No referenced tuple with every part non-NULL repeats, and the referencing tuples are
    held at the containment floor, on the column lists in pair order. A count left undecided
    leaves the join open, so it does not recheck."""
    columns = tuple(p["right_column"] for p in predicates)
    parent_columns = tuple(p["left_column"] for p in predicates)
    rows, null_rows = counts.rows(parent), counts.null_rows(parent, parent_columns)
    distinct = counts.evidence.contained(parent, parent_columns, parent, parent_columns)
    held = counts.evidence.contained(child, columns, parent, parent_columns)
    if rows is None or null_rows is None or distinct is None or held is None:
        logger.warning("Join recheck on {} -> {} left open: a count is undecided", child, parent)
        return False
    repeated = rows - null_rows - distinct[0]
    logger.info("Join recheck on {} -> {}: {} orphan pairs, {} repeated referenced tuples",
                child, parent, held[0] - held[1], repeated)
    return repeated == 0 and counts.within(child, columns, parent, parent_columns)


def read_oracle(path: Path, kinds: Iterable[str]) -> dict[str, Cells]:
    """Claims per fact kind from an oracle file in the contract's JSON Lines format."""
    known = set(kinds)
    claims: dict[str, Cells] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        claim = json.loads(line)
        if claim["kind"] not in known:
            raise ValueError(f"{path}:{number}: unknown fact kind {claim['kind']!r}")
        claims.setdefault(claim["kind"], {})[tuple(claim["key"])] = claim["value"]
    return claims
