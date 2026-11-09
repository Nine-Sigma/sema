"""Statements, types and workload parsing for a profile's scoped library projection."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from showcase.ontology_query_benchmark.profile import Extract

DUCKDB_TYPES = {
    "string": "VARCHAR",
    "boolean": "BOOLEAN",
    "timestamp": "TIMESTAMPTZ",
    "array<string>": "VARCHAR[]",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declared_de_ids(spec_paths: Sequence[Path], id_pattern: str) -> tuple[str, ...]:
    ids: set[str] = set()
    for path in spec_paths:
        fields = json.loads(path.read_text(encoding="utf-8"))["fields"]
        ids.update(str(field["de_id"]) for field in fields if field.get("de_id"))
    malformed = sorted(i for i in ids if not re.fullmatch(id_pattern, i))
    if malformed:
        raise ValueError(f"declared de_id values do not match {id_pattern!r}: {malformed}")
    return tuple(sorted(ids))


def duckdb_type(warehouse_type: str) -> str:
    if warehouse_type.startswith("varchar("):
        return "VARCHAR"
    if warehouse_type not in DUCKDB_TYPES:
        raise ValueError(f"no DuckDB type for warehouse type {warehouse_type!r}")
    return DUCKDB_TYPES[warehouse_type]


def plain(value: Any) -> Any:
    """The SQL connector returns arrays as numpy arrays; DuckDB takes lists."""
    return value.tolist() if hasattr(value, "tolist") else value


def qualified(spec: Extract, relation: str) -> str:
    return f"{spec.catalog}.{spec.schema_name}.{relation}"


def _names(relations: Sequence[str]) -> str:
    return ", ".join(f"'{name}'" for name in relations)


def columns_sql(spec: Extract) -> str:
    return (
        "SELECT table_name, column_name, full_data_type"
        f" FROM {spec.catalog}.information_schema.columns"
        f" WHERE table_schema = '{spec.schema_name}' AND table_name IN ({_names(spec.relations)})"
        " ORDER BY table_name, ordinal_position"
    )


def tables_sql(spec: Extract) -> str:
    return (
        f"SELECT table_name, last_altered, comment FROM {spec.catalog}.information_schema.tables"
        f" WHERE table_schema = '{spec.schema_name}' AND table_name IN ({_names(spec.relations)})"
        " ORDER BY table_name"
    )


def predicates(spec: Extract, de_ids: Sequence[str]) -> dict[str, str]:
    """Element rows by declared id; member rows by an element's member key or a declared id."""
    ids, elements, members = _names(de_ids), spec.elements, spec.members
    return {
        elements.relation: f"{elements.key} IN ({ids})",
        members.relation: (
            f"{members.key} IN (SELECT {elements.member_key} FROM"
            f" {qualified(spec, elements.relation)} WHERE {elements.key} IN ({ids}))"
            f" OR {members.key} IN ({ids})"
        ),
    }


def count_sql(spec: Extract, relation: str, predicate: str) -> str:
    return f"SELECT COUNT(*) FROM {qualified(spec, relation)} WHERE {predicate}"


def select_sql(
    spec: Extract, relation: str, columns: Sequence[str], predicate: str, row_limit: int
) -> str:
    listed = ", ".join(f"`{name}`" for name in columns)
    order = ", ".join(f"`{name}`" for name in order_keys(spec, relation))
    return (
        f"SELECT {listed} FROM {qualified(spec, relation)} WHERE {predicate}"
        f" ORDER BY {order} LIMIT {row_limit + 1}"
    )


def order_keys(spec: Extract, relation: str) -> tuple[str, ...]:
    return next(r.order_keys for r in (spec.elements, spec.members) if r.relation == relation)
