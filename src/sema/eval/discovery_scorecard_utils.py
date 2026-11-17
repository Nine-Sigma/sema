"""Reads behind the discovery scorecard: graph joins, declared keys, data rechecks, oracle files.

Keys follow the oracle file format: relation and column names lowercased, a column list as its
sorted names joined by `+`. Every warehouse read is a Databricks-dialect statement run through
`Connector.execute_query`, which each connector translates to its own dialect.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from sema.connectors.base import Connector
from sema.connectors.databricks_utils import METADATA_SOURCE, sql_string_literal
from sema.intake.run.scorecard import Cells, Key
from sema.log import logger
from sema.models.extraction import ExtractedForeignKey

COLUMN_SEPARATOR = "+"
# Map access, so a graph where no join has a cardinality yet raises no unknown-key warning.
GRAPH_JOINS = (
    "MATCH (jp:JoinPath) "
    "RETURN jp.join_predicates AS predicates, properties(jp).cardinality_hint AS cardinality"
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


def unique_keys_query(catalog: str, schema: str) -> str:
    """Each column of every primary key and unique constraint declared in one schema."""
    same_constraint = " AND ".join(
        f"kcu.constraint_{part} = tc.constraint_{part}" for part in ("catalog", "schema", "name")
    )
    return (
        "SELECT tc.table_name, tc.constraint_name, kcu.column_name "
        f"FROM {METADATA_SOURCE}.`table_constraints` tc "
        f"JOIN {METADATA_SOURCE}.`key_column_usage` kcu ON {same_constraint} "
        "AND kcu.table_name = tc.table_name "
        "WHERE tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE') "
        f"AND tc.table_catalog = {sql_string_literal(catalog)} "
        f"AND tc.table_schema = {sql_string_literal(schema)}"
    )


def unique_keys(rows: Iterable[tuple[Any, ...]]) -> set[Key]:
    """(relation, columns) per declared unique constraint, from `unique_keys_query` rows."""
    columns: dict[tuple[str, str], list[str]] = {}
    for table, constraint, column in rows:
        columns.setdefault((str(table), str(constraint)), []).append(str(column))
    return {(table.lower(), columns_key(names)) for (table, _), names in columns.items()}


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
) -> bool:
    """Whether every referencing value of a one-column join occurs on the referenced side.

    shortcut: a composite key's columns are stored sorted, not paired, so a composite join is
    never rechecked; upgrade when the fact keeps its column pairs.
    """
    relation, columns, referenced, referenced_columns = key
    if COLUMN_SEPARATOR in columns or relation not in tables or referenced not in tables:
        return False
    query = orphan_count_query(
        catalog, (*tables[relation], columns), (*tables[referenced], referenced_columns),
    )
    try:
        rows = connector.execute_query(query)
    except Exception as exc:  # each connector raises its own driver's error type
        logger.warning("Join recheck on {} failed: {}", key, exc)
        return False
    return bool(rows) and int(rows[0][0]) == 0


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
