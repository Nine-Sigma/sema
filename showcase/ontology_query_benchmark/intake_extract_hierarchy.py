"""A bounded concept-hierarchy slice, each relation selected by membership against ids
already extracted: mapping rows of the workload members, the concepts and ancestors
their targets reach, and the parent edges among those ancestors. The closure itself is
never copied; executed SQL reads it in the warehouse."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from sema.connectors.databricks_utils import sql_string_literal
from showcase.ontology_query_benchmark.intake_extract_utils import duckdb_type
from showcase.ontology_query_benchmark.profile import ConceptHierarchy

SCHEMA = "concept_hierarchy"
Rows = Callable[[str], list[tuple[Any, ...]]]


@dataclass(frozen=True)
class Slice:
    name: str
    source: str
    columns: tuple[tuple[str, str, str], ...]
    updated_at: str
    comment: str | None
    expected: int
    rows: tuple[tuple[Any, ...], ...] | None

    def values(self, position: int) -> set[str]:
        return {str(row[position]) for row in self.rows or () if row[position] is not None}


def read_hierarchy(
    rows: Rows, spec: ConceptHierarchy, members: set[str], row_limit: int
) -> tuple[list[Slice], list[str]]:
    before = {r: _altered(rows, r) for r in spec.relations}
    slices: list[Slice] = []
    targets = members
    if (m := spec.mapping) is not None:
        mapped = f"{m.type_column} IN ({_in(m.mapping_types)})"
        first = f"{_member(m.source_column, members)} AND {mapped}"
        replaced = (
            f" OR ({m.type_column} IN ({_in(m.replacement_types)}) AND {m.source_column} IN"
            f" (SELECT {m.target_column} FROM {m.relation} WHERE {first}))"
            if m.replacement_types else ""
        )
        columns = (m.source_column, m.target_column, m.type_column)
        slices.append(_read(rows, "mapping", m.relation, columns, f"({first}){replaced}",
                            row_limit, before))  # fmt: skip
        targets = slices[-1].values(1)
    c, k = spec.closure, spec.concepts
    ancestors = (f"IN (SELECT {c.ancestor_column} FROM {c.relation}"
                 f" WHERE {_member(c.descendant_column, targets)})")  # fmt: skip
    named = (k.id_column, k.name_column, *((k.standard_column,) if k.standard_column else ()))
    concept = f"{_member(k.id_column, targets)} OR {k.id_column} {ancestors}"
    slices.append(_read(rows, "concept", k.relation, named, concept, row_limit, before))
    des = c.descendant_column
    edge = f"{c.level_column} = 1 AND ({_member(des, targets)} OR {des} {ancestors})"
    slices.append(_read(rows, "edge", c.relation, (c.ancestor_column, des), edge,
                        row_limit, before))  # fmt: skip
    moved = [
        f"{SCHEMA}.{s.name}: updated_at moved from {before[s.source][0]} to {after}"
        for s in slices if (after := _altered(rows, s.source)[0]) != before[s.source][0]
    ]
    return slices, [*_short(slices, row_limit), *moved]


def _read(
    rows: Rows, name: str, relation: str, columns: Sequence[str], predicate: str,
    row_limit: int, before: dict[str, tuple[str, str | None]],
) -> Slice:  # fmt: skip
    listed = ", ".join(columns)
    distinct = f"SELECT DISTINCT {listed} FROM {relation} WHERE {predicate}"
    expected = int(rows(f"SELECT COUNT(*) FROM ({distinct})")[0][0])
    read = None
    if expected <= row_limit:
        read = tuple(rows(f"{distinct} ORDER BY {listed} LIMIT {row_limit + 1}"))
    types = dict(_types(rows, relation))
    described = tuple((col, types[col], duckdb_type(types[col])) for col in columns)
    when, comment = before[relation]
    return Slice(name, relation, described, when, comment, expected, read)


def _short(slices: Iterable[Slice], row_limit: int) -> list[str]:
    found = []
    for s in slices:
        if s.rows is None:
            found.append(
                f"{SCHEMA}.{s.name}: {s.expected} counted rows exceed row_limit {row_limit}"
            )
        elif len(s.rows) != s.expected:
            found.append(f"{SCHEMA}.{s.name}: returned_rows {len(s.rows)} != counted {s.expected}")
    return found


def _types(rows: Rows, relation: str) -> list[tuple[str, str]]:
    catalog, schema, table = relation.split(".")
    found = rows(
        f"SELECT table_name, column_name, full_data_type FROM {catalog}.information_schema.columns"
        f" WHERE table_schema = '{schema}' AND table_name = '{table}' ORDER BY ordinal_position"
    )
    return [(str(column), str(kind)) for _, column, kind in found]


def _altered(rows: Rows, relation: str) -> tuple[str, str | None]:
    catalog, schema, table = relation.split(".")
    found = rows(
        f"SELECT table_name, last_altered, comment FROM {catalog}.information_schema.tables"
        f" WHERE table_schema = '{schema}' AND table_name = '{table}'"
    )
    if not found:
        raise ValueError(f"hierarchy relation not in the catalog: {relation}")
    when = found[0][1]
    return (when.isoformat() if hasattr(when, "isoformat") else str(when)), found[0][2]


def _member(column: str, ids: Iterable[str]) -> str:
    """Ids compare as text, so an id column of any type takes ids read from another."""
    return f"CAST({column} AS STRING) IN ({_in(ids)})"


def _in(values: Iterable[str]) -> str:
    """An empty set renders NULL, which matches no row."""
    return ", ".join(sql_string_literal(str(v)) for v in sorted(values)) or "NULL"
