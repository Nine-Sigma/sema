"""Helpers for the FK / join detector.

Pure-functional name-affinity matching, type compatibility, and
cardinality verification. The detector itself orchestrates these
helpers; the helpers never touch I/O.

Coded columns: a column of at most `LOW_CARDINALITY_BOUND` distinct values
is nominated against a code relation, one whose key column is unique,
non-null and at most `LOW_CARDINALITY_BOUND` rows. The data shape alone
nominates against a key with a text column beside it; equal normalized
names nominate against any such key. `containment` decides: a value set
must list at least `COVERAGE_THRESHOLD` of the column's distinct non-null
values and of its non-null rows, the share the detector asks of a foreign
key (a code list may miss a legacy code, never most of the column). The
bound is the profiler's categorical threshold: past it a column is an
identifier or a measure, and a relation an entity, not a code list.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

from sema.models.config import ProfilingConfig
from sema.models.extraction import ExtractedColumn

_INTEGER_TYPES = frozenset({
    "int", "integer", "bigint", "smallint", "tinyint", "long",
})
_TEXT_TYPES = frozenset({
    "string", "varchar", "char", "text", "uuid",
})
_FK_NAME_SUFFIX_RE = re.compile(r"^(?P<entity>.+?)_(id|key|code)$")
COVERAGE_THRESHOLD = 0.80
LOW_CARDINALITY_BOUND: int = ProfilingConfig.model_fields["categorical_threshold"].default
Profiles = Mapping[tuple[str, str, str], tuple[int, int]]


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
            out.append(fk_candidate(pk_col, fk_col))
    return out


def fk_candidate(pk_col: ExtractedColumn, fk_col: ExtractedColumn) -> FKCandidate:
    return FKCandidate(
        pk_table=pk_col.table_name,
        pk_column=pk_col.name,
        fk_table=fk_col.table_name,
        fk_column=fk_col.name,
        pk_type=pk_col.data_type,
        fk_type=fk_col.data_type,
        schema_name=fk_col.schema,
        pk_ref=pk_col.ref or "",
        fk_ref=fk_col.ref or "",
        catalog=fk_col.catalog,
    )


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


def normalized_name(name: str) -> str:
    """Lowercase, each run of other characters one `_`, no `_` at either end."""
    return re.sub(r"[\W_]+", "_", name.lower()).strip("_")


def is_text(column: ExtractedColumn) -> bool:
    return normalize_type(column.data_type) in _TEXT_TYPES


def column_key(column: ExtractedColumn) -> tuple[str, str, str]:
    return (column.schema, column.table_name, column.name)


@dataclass(frozen=True)
class Containment:
    """How much of a column's data a value set lists; shares are None for a column of NULLs."""
    observed_distinct: int
    null_rows: int
    listed_value_share: float | None
    listed_row_share: float | None

    @property
    def holds(self) -> bool:
        shares = (self.listed_value_share, self.listed_row_share)
        return all(share is not None and share >= COVERAGE_THRESHOLD for share in shares)


def containment(
    counts: Mapping[Any, int], listed: Collection[Any],
) -> Containment:
    """`counts` maps each observed value, NULL as None, to its rows."""
    null_rows = counts.get(None, 0)
    observed = {value: rows for value, rows in counts.items() if value is not None}
    if not observed:
        return Containment(0, null_rows, None, None)
    listed_rows = [rows for value, rows in observed.items() if value in listed]
    return Containment(
        observed_distinct=len(observed),
        null_rows=null_rows,
        listed_value_share=len(listed_rows) / len(observed),
        listed_row_share=sum(listed_rows) / sum(observed.values()),
    )


def is_code_key(stats: tuple[int, int] | None, bound: int) -> bool:
    """Unique and non-null (distinct equals rows) in a relation of at most `bound` rows."""
    return stats is not None and 0 < stats[0] == stats[1] <= bound


def _coded_pairs(
    columns: list[ExtractedColumn],
    profiles: Profiles,
    bound: int,
    keys: list[ExtractedColumn],
    pairs: Callable[[ExtractedColumn, ExtractedColumn], bool],
) -> list[FKCandidate]:
    out: list[FKCandidate] = []
    for col in columns:
        stats = profiles.get(column_key(col))
        if stats is None or not 0 < stats[0] <= bound or col.ref is None:
            continue
        for key in keys:
            if key.schema != col.schema or key.table_name == col.table_name:
                continue
            if types_compatible(key.data_type, col.data_type) and pairs(col, key):
                out.append(fk_candidate(key, col))
    return out


def _code_keys(
    columns: list[ExtractedColumn], profiles: Profiles, bound: int,
) -> list[ExtractedColumn]:
    return [
        c for c in columns
        if c.ref is not None and is_code_key(profiles.get(column_key(c)), bound)
    ]


def nominate_by_data_shape(
    columns: list[ExtractedColumn], profiles: Profiles, bound: int = LOW_CARDINALITY_BOUND,
) -> list[FKCandidate]:
    """Low-distinct columns against code keys that carry a text column beside them; no name."""
    texts = {(c.schema, c.table_name, c.name) for c in columns if is_text(c)}
    keys = [
        k for k in _code_keys(columns, profiles, bound)
        if any(s == k.schema and t == k.table_name and n != k.name for s, t, n in texts)
    ]
    return _coded_pairs(columns, profiles, bound, keys, lambda _col, _key: True)


def nominate_by_name(
    columns: list[ExtractedColumn], profiles: Profiles, bound: int = LOW_CARDINALITY_BOUND,
) -> list[FKCandidate]:
    """Low-distinct columns against code keys of an equal normalized name."""
    return _coded_pairs(
        columns, profiles, bound, _code_keys(columns, profiles, bound),
        lambda col, key: normalized_name(col.name) == normalized_name(key.name),
    )
