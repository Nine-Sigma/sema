"""Helpers for the FK / join detector.

Pure-functional name-affinity matching, type compatibility, and
cardinality verification. The detector itself orchestrates these
helpers; the helpers never touch I/O.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sema.models.extraction import ExtractedColumn

_INTEGER_TYPES = frozenset({
    "int", "integer", "bigint", "smallint", "tinyint", "long",
})
_TEXT_TYPES = frozenset({
    "string", "varchar", "char", "text", "uuid",
})
_FK_NAME_SUFFIX_RE = re.compile(r"^(?P<entity>.+?)_(id|key|code)$")


@dataclass(frozen=True)
class FKCandidate:
    """A potential foreign-key relationship between two columns.

    PK / FK semantics: the FK column REFERENCES the PK column.
    Both columns are constrained to the same `schema_name`.
    """
    pk_table: str
    pk_column: str
    fk_table: str
    fk_column: str
    pk_type: str
    fk_type: str
    schema_name: str
    pk_ref: str
    fk_ref: str
    catalog: str = ""


def normalize_type(data_type: str) -> str:
    """Lowercase, strip parameter list (e.g., `VARCHAR(64)` → `varchar`)."""
    base = data_type.split("(", 1)[0].strip().lower()
    return base


def types_compatible(left: str, right: str) -> bool:
    a = normalize_type(left)
    b = normalize_type(right)
    if a == b:
        return True
    if a in _INTEGER_TYPES and b in _INTEGER_TYPES:
        return True
    if a in _TEXT_TYPES and b in _TEXT_TYPES:
        return True
    return False


def fk_name_root(column_name: str) -> str | None:
    """Return the entity root from a FK-style column name.

    `patient_id` → `patient`, `sample_key` → `sample`, `gene_code` → `gene`.
    Returns None when the name does not match the FK suffix pattern.
    """
    match = _FK_NAME_SUFFIX_RE.match(column_name.lower())
    return match.group("entity") if match else None


def has_name_affinity(
    fk_name: str, pk_col: ExtractedColumn, fk_root: str,
) -> bool:
    """Names propose a pair; data evidence decides it.

    Equal names, a role prefix before a key-shaped referenced name
    (`from_<x>_id` -> `<x>_id`), or a referenced table named after the
    referencing root. A bare `id` is not key-shaped, so it never pairs
    by prefix with every `<x>_id`.
    """
    fk = fk_name.lower()
    pk = pk_col.name.lower()
    if fk == pk:
        return True
    if fk.endswith(f"_{pk}") and fk_name_root(pk) is not None:
        return True
    return pk_col.table_name.lower() == fk_root


def enumerate_candidates_from_metadata(
    columns: list[ExtractedColumn],
) -> list[FKCandidate]:
    """Enumerate intra-schema FK candidates by name affinity + type."""
    by_schema: dict[str, list[ExtractedColumn]] = {}
    for col in columns:
        by_schema.setdefault(col.schema, []).append(col)

    candidates: list[FKCandidate] = []
    for schema, cols in by_schema.items():
        candidates.extend(_candidates_within_schema(cols))
    return candidates


def _candidates_within_schema(
    cols: list[ExtractedColumn],
) -> list[FKCandidate]:
    out: list[FKCandidate] = []
    for fk_col in cols:
        root = fk_name_root(fk_col.name)
        if not root or fk_col.ref is None:
            continue
        if fk_col.table_name.lower() == root:
            continue
        for pk_col in cols:
            if pk_col is fk_col or pk_col.ref is None:
                continue
            if pk_col.table_name == fk_col.table_name:
                continue
            if not has_name_affinity(fk_col.name, pk_col, root):
                continue
            if not types_compatible(pk_col.data_type, fk_col.data_type):
                continue
            out.append(FKCandidate(
                pk_table=pk_col.table_name,
                pk_column=pk_col.name,
                fk_table=fk_col.table_name,
                fk_column=fk_col.name,
                pk_type=pk_col.data_type,
                fk_type=fk_col.data_type,
                schema_name=fk_col.schema,
                pk_ref=pk_col.ref,
                fk_ref=fk_col.ref,
                catalog=fk_col.catalog,
            ))
    return out


def verify_cardinality(
    pk_distinct: int,
    pk_rows: int,
    fk_distinct: int,
) -> bool:
    """PK uniquely valued AND FK distinct count ≤ PK distinct count."""
    if pk_distinct <= 0 or pk_rows <= 0:
        return False
    if pk_distinct != pk_rows:
        return False
    return fk_distinct <= pk_distinct
