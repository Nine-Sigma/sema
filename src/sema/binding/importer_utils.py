"""Helpers for the concept source importer: SQL text, row decoding, crosswalks.

SQL is built from spec identifiers and spec filter fragments only, never from
row values, so a library value can never change the statement.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from sema.binding.spec import (
    ConceptSourceSpec,
    DetailsSpec,
    EndpointRole,
    ExternalRefRole,
)

QueryFn = Callable[[str], list[tuple[Any, ...]]]
Record = dict[str, Any]
RefKey = tuple[str, str]

BOOL_TEXT = {"true": True, "false": False, "1": True, "0": False}


@dataclass
class Details:
    definition: str | None = None
    external_refs: set[RefKey] = field(default_factory=set)


def select_sql(relation: str, columns: Iterable[str], where: str | None = None) -> str:
    sql = f"SELECT {', '.join(columns)} FROM {relation}"
    return f"{sql} WHERE {where}" if where else sql


def concept_scope_sql(spec: ConceptSourceSpec) -> str:
    concepts = spec.concepts
    if concepts is None:
        raise ValueError(f"spec {spec.id} has no concepts section to scope by")
    return select_sql(
        concepts.relation.name, [concepts.roles.concept_id], concepts.filter
    )


def in_concept_scope(column: str, spec: ConceptSourceSpec) -> str:
    return f"{column} IN ({concept_scope_sql(spec)})"


def fetch(
    query: QueryFn,
    relation: str,
    columns: Iterable[str | None],
    where: str | None = None,
) -> list[Record]:
    """Rows as column-keyed dicts, sorted so the import is order-independent."""
    names = list(dict.fromkeys(c for c in columns if c))
    records = [
        dict(zip(names, row)) for row in query(select_sql(relation, names, where))
    ]
    return sorted(records, key=lambda r: [str(r[name]) for name in names])


def as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def expand_list(value: Any) -> list[str]:
    """ARRAY values, JSON or Python-literal list strings, or one string, verbatim.

    Entries are never split on commas: a column entry can be a SQL expression.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return _expand_string(value)
    if callable(getattr(value, "tolist", None)):
        # Databricks returns ARRAY columns as numpy arrays; src avoids a numpy import
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        text = as_text(value)
        return [text] if text else []
    items = (as_text(item) for item in value)
    return [item for item in items if item]


def _expand_string(value: str) -> list[str]:
    text = value.strip()
    if not text.startswith("["):
        return [text] if text else []
    decoded = _decode_list(text)
    return expand_list(decoded) if decoded is not None else [text]


def _decode_list(text: str) -> list[Any] | None:
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = _literal_str_list(text)
    return decoded if isinstance(decoded, list) else None


def _literal_str_list(text: str) -> list[str] | None:
    try:
        decoded = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    if isinstance(decoded, list) and all(isinstance(item, str) for item in decoded):
        return decoded
    return None


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return {1: True, 0: False}.get(value)
    if not isinstance(value, str):
        return None
    return BOOL_TEXT.get(value.strip().lower())


def ref_columns(roles: list[ExternalRefRole]) -> list[str]:
    columns: list[str] = []
    for role in roles:
        columns.append(role.column)
        if _is_template(role.system):
            columns.append(role.system[1:-1])
    return columns


def row_external_refs(row: Record, roles: list[ExternalRefRole]) -> set[RefKey]:
    refs: set[RefKey] = set()
    for role in roles:
        system_value = (
            row[role.system[1:-1]] if _is_template(role.system) else role.system
        )
        system, ref_id = as_text(system_value), as_text(row[role.column])
        if system and ref_id:
            refs.add((system, ref_id))
    return refs


def _is_template(system: str) -> bool:
    return system.startswith("{") and system.endswith("}")


def read_details(
    query: QueryFn, details: DetailsSpec, source_sql: str
) -> dict[str, Details]:
    """Details keyed by the join value, limited to keys the source relation holds."""
    key_column = details.on[1]
    rows = fetch(
        query,
        details.relation.name,
        [
            key_column,
            details.roles.definition,
            *ref_columns(details.roles.external_refs),
        ],
        f"{key_column} IN ({source_sql})",
    )
    index: dict[str, Details] = {}
    for row in rows:
        key = as_text(row[key_column])
        if key is None:
            continue
        entry = index.setdefault(key, Details())
        if entry.definition is None and details.roles.definition:
            entry.definition = as_text(row[details.roles.definition])
        entry.external_refs |= row_external_refs(row, details.roles.external_refs)
    return index


def endpoint(row: Record, role: EndpointRole) -> RefKey | None:
    ref_id = as_text(row[role.id])
    if ref_id:
        return role.system, ref_id
    if role.fallback is None:
        return None
    fallback_id = as_text(row[role.fallback.id])
    return (role.fallback.system, fallback_id) if fallback_id else None


def endpoint_columns(role: EndpointRole) -> list[str]:
    return [role.id] + ([role.fallback.id] if role.fallback else [])


def refs_payload(refs: Iterable[RefKey]) -> list[dict[str, str]]:
    return [{"system": system, "id": ref_id} for system, ref_id in sorted(set(refs))]


def group_by_key(rows: list[Record], column: str) -> dict[str, list[Record]]:
    groups: dict[str, list[Record]] = {}
    for row in rows:
        key = as_text(row[column])
        if key is not None:
            groups.setdefault(key, []).append(row)
    return groups


def first_text(rows: list[Record], column: str | None) -> str | None:
    if column is None:
        return None
    return next((text for row in rows if (text := as_text(row[column]))), None)


def synonyms_of(
    rows: list[Record], column: str | None, extra: Iterable[str | None]
) -> set[str]:
    found = {s for row in rows for s in expand_list(row[column])} if column else set()
    return found | {s for s in extra if s}


def matched_details(
    rows: list[Record], column: str | None, index: dict[str, Details]
) -> list[Details]:
    if column is None:
        return []
    keys = sorted({key for row in rows if (key := as_text(row[column])) is not None})
    return [index[key] for key in keys if key in index]
