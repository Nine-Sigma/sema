"""Statements, types and workload parsing for the Standard Library workload projection."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CATALOG = "partner_library_test"
SCHEMA = "standard_library"
DATA_ELEMENT = "standard_library_data_element"
CONCEPT = "standard_library_concept"
RELATIONS = (DATA_ELEMENT, CONCEPT)
ORDER_KEYS = {
    DATA_ELEMENT: (
        "data_element_prt_id", "permissible_value_prt_id", "data_element_enriched_surrogate_id",
    ),
    CONCEPT: ("concept_prt_id",),
}
DUCKDB_TYPES = {
    "string": "VARCHAR",
    "boolean": "BOOLEAN",
    "timestamp": "TIMESTAMPTZ",
    "array<string>": "VARCHAR[]",
}
DE_ID = re.compile(r"^[A-Z0-9]+$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declared_de_ids(spec_paths: Sequence[Path]) -> tuple[str, ...]:
    ids: set[str] = set()
    for path in spec_paths:
        fields = json.loads(path.read_text(encoding="utf-8"))["fields"]
        ids.update(str(field["de_id"]) for field in fields if field.get("de_id"))
    malformed = sorted(i for i in ids if not DE_ID.match(i))
    if malformed:
        raise ValueError(f"declared de_id values are not partner ids: {malformed}")
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


def qualified(relation: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{relation}"


def _names(relations: Sequence[str]) -> str:
    return ", ".join(f"'{name}'" for name in relations)


def columns_sql() -> str:
    return (
        f"SELECT table_name, column_name, full_data_type FROM {CATALOG}.information_schema.columns"
        f" WHERE table_schema = '{SCHEMA}' AND table_name IN ({_names(RELATIONS)})"
        " ORDER BY table_name, ordinal_position"
    )


def tables_sql() -> str:
    return (
        f"SELECT table_name, last_altered, comment FROM {CATALOG}.information_schema.tables"
        f" WHERE table_schema = '{SCHEMA}' AND table_name IN ({_names(RELATIONS)})"
        " ORDER BY table_name"
    )


def predicates(de_ids: Sequence[str]) -> dict[str, str]:
    ids = _names(de_ids)
    return {
        DATA_ELEMENT: f"data_element_prt_id IN ({ids})",
        CONCEPT: (
            f"concept_prt_id IN (SELECT permissible_value_prt_id FROM {qualified(DATA_ELEMENT)}"
            f" WHERE data_element_prt_id IN ({ids})) OR concept_prt_id IN ({ids})"
        ),
    }


def count_sql(relation: str, predicate: str) -> str:
    return f"SELECT COUNT(*) FROM {qualified(relation)} WHERE {predicate}"


def select_sql(relation: str, columns: Sequence[str], predicate: str, row_limit: int) -> str:
    listed = ", ".join(f"`{name}`" for name in columns)
    order = ", ".join(f"`{name}`" for name in ORDER_KEYS[relation])
    return (
        f"SELECT {listed} FROM {qualified(relation)} WHERE {predicate}"
        f" ORDER BY {order} LIMIT {row_limit + 1}"
    )
