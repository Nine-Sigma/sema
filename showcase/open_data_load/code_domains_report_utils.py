"""The bindings SEMA proposes for coded columns, placed in a warehouse, and their data coverage.

A spec's stated domain names a dictionary entry, a property and a vocabulary. The entry
names its table when every token of the table's name appears in the entry's name (the most
tokens win; a tie is ambiguous); the property names the column whose name tokens equal its
own; a vocabulary's code relation is the table whose name tokens equal the code list's
relation name, and its code column the one equal to the code role.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import CodeListSpec
from sema.connectors.duckdb import DuckDBConnector

STATED_DOMAIN = "stated_domain"
DECLARED_KEY = "declared_key"
BUILD_GRAPH = "build_graph"


@dataclass(frozen=True)
class WarehouseTable:
    name: str
    ref: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class CodeSource:
    table: str
    column: str


@dataclass(frozen=True)
class Proposal:
    table: str | None
    column: str | None
    value_set: str
    codes_at: CodeSource | None
    evidence: frozenset[str]
    problems: tuple[str, ...] = ()


@dataclass(frozen=True)
class Coverage:
    null_rows: int
    listed_value_share: float | None = None
    listed_row_share: float | None = None


@dataclass
class _Merged:
    proposal: Proposal
    evidence: set[str] = field(default_factory=set)
    problems: list[str] = field(default_factory=list)


def name_tokens(name: str) -> frozenset[str]:
    return frozenset(re.findall(r"[0-9a-z]+", name.casefold()))


def unqualified(relation: str) -> str:
    return relation.rsplit(".", 1)[-1]


def _equal_name(name: str, candidates: Iterable[str]) -> str | None:
    wanted = name_tokens(name)
    return next((c for c in candidates if name_tokens(c) == wanted), None)


def place_entry(entry_name: str, tables: Mapping[str, WarehouseTable]) -> tuple[str | None, str]:
    """The table a dictionary entry names, or None and why not."""
    wanted = name_tokens(entry_name)
    fits = [name for name in tables if name_tokens(name) <= wanted]
    if not fits:
        return None, f"no warehouse table matches dictionary entry {entry_name!r}"
    most = max(len(name_tokens(name)) for name in fits)
    best = sorted(name for name in fits if len(name_tokens(name)) == most)
    if len(best) > 1:
        return None, f"dictionary entry {entry_name!r} matches tables {', '.join(best)}"
    return best[0], ""


def code_source(
    code_list: CodeListSpec | None, tables: Mapping[str, WarehouseTable],
) -> CodeSource | None:
    if code_list is None:
        return None
    table = _equal_name(unqualified(code_list.relation.name), tables)
    column = _equal_name(code_list.roles.code, tables[table].columns) if table else None
    return CodeSource(table, column) if table and column else None


def stated_domains(
    spec: ConceptSourceSpec, tables: Mapping[str, WarehouseTable],
) -> list[Proposal]:
    code_lists = {entry.vocabulary_id: entry for entry in spec.code_lists or []}
    model = spec.entity_model
    proposals = []
    for entry in (model.entities or []) if model else []:
        entry_name = entry.entity or unqualified(entry.relation.name)
        table, unplaced = place_entry(entry_name, tables)
        for prop, vocabulary in entry.domains.items():
            column = _equal_name(prop, tables[table].columns) if table else None
            codes_at = code_source(code_lists.get(vocabulary), tables)
            problems = [unplaced] if table is None else []
            if table and column is None:
                problems.append(f"no column of {table} matches stated property {prop!r}")
            if codes_at is None:
                problems.append(f"value set {vocabulary} has no code relation in the warehouse")
            proposals.append(Proposal(
                table, column, vocabulary, codes_at, frozenset({STATED_DOMAIN}), tuple(problems),
            ))  # fmt: skip
    return proposals


def declared_keys(
    connector: DuckDBConnector, catalog: str, schema: str, tables: Iterable[str],
) -> list[Proposal]:
    """Single-column declared keys; composite keys are out of scope."""
    proposals = []
    for table in tables:
        for key in connector.extract_foreign_keys_dto(catalog, schema, table):
            if len(key.column_pairs) != 1:
                continue
            ((referencing, referenced),) = key.column_pairs
            proposals.append(Proposal(
                table, referencing, key.referenced_table,
                CodeSource(key.referenced_table, referenced), frozenset({DECLARED_KEY}),
            ))  # fmt: skip
    return proposals


def merged(proposals: Iterable[Proposal]) -> list[Proposal]:
    """One proposal per placed column and value-set source; unplaced ones stay apart."""
    by_key: dict[object, _Merged] = {}
    for index, proposal in enumerate(proposals):
        target = proposal.codes_at or proposal.value_set
        key = (proposal.table, proposal.column, target) if proposal.column else index
        entry = by_key.setdefault(key, _Merged(proposal))
        entry.evidence |= proposal.evidence
        entry.problems += [p for p in proposal.problems if p not in entry.problems]
    return [
        Proposal(m.proposal.table, m.proposal.column, m.proposal.value_set,
                 m.proposal.codes_at, frozenset(m.evidence), tuple(m.problems))
        for m in by_key.values()
    ]  # fmt: skip


def _ident(*parts: str) -> str:
    return ".".join("`" + part.replace("`", "``") + "`" for part in parts)


def code_values(connector: DuckDBConnector, catalog: str, schema: str, at: CodeSource) -> set[str]:
    column = _ident(at.column)
    rows = connector.execute_query(
        f"SELECT DISTINCT CAST({column} AS STRING) FROM {_ident(catalog, schema, at.table)} "
        f"WHERE {column} IS NOT NULL"
    )
    return {row[0] for row in rows}


def coverage(
    connector: DuckDBConnector, catalog: str, schema: str, table: str, column: str,
    codes: set[str] | frozenset[str] | None,
) -> Coverage:
    # shortcut: distinct values are counted in memory; push the IN test into SQL if a bound
    # column is ever high-cardinality.
    rows = connector.execute_query(
        f"SELECT CAST({_ident(column)} AS STRING), count(*) "
        f"FROM {_ident(catalog, schema, table)} GROUP BY 1"
    )
    counts = {value: int(n) for value, n in rows}
    null_rows = counts.pop(None, 0)
    if codes is None or not counts:
        return Coverage(null_rows)
    listed = {value: n for value, n in counts.items() if value in codes}
    return Coverage(
        null_rows,
        listed_value_share=len(listed) / len(counts),
        listed_row_share=sum(listed.values()) / sum(counts.values()),
    )
