"""US-002 / G-01: the gold set's executable source specification.

A gold set is a snapshot of a **declared** scope. Auto-discovering every
``cbioportal_*`` schema made the scope a function of whatever had been ingested,
so a new study silently changed the denominator and reddened the suite.

A schema list alone cannot express the declaration, because the two sources are
different shapes: raw cbioportal studies are one ``sample`` table per schema,
while ``sema_staging.condition_staging`` is one table whose studies are a
``source_schema`` **column value**. :class:`SourceSpec` is therefore executable —
it carries the table, the code column, and how the scope is addressed — and one
enumerator per :class:`SourceKind` renders it.

Discovery survives here for the refresh path (proposing a new scope to a human),
never for enumeration at test time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

__all__ = [
    "MainTypeReport",
    "SourceKind",
    "SourceSpec",
    "discover_oncotree_schemas",
    "enumerate_scoped_codes",
    "raw_samples_view",
    "scoped_enumeration_sql",
    "source_main_types",
]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


class _Cursor(Protocol):
    def execute(self, sql: str) -> Any: ...
    def fetchall(self) -> list[Any]: ...


class SourceKind(str, Enum):
    """How a declared scope is addressed in SQL."""

    RAW_SAMPLES = "raw_samples"
    STAGING = "staging"


@dataclass(frozen=True)
class SourceSpec:
    """An executable declaration of what the gold set is a snapshot of.

    ``RAW_SAMPLES``: ``scope_values`` are schemas, each holding ``table``.
    ``STAGING``: ``table`` is fully qualified and ``scope_column`` holds the
    study, so ``scope_values`` are filter values.
    """

    kind: SourceKind
    table: str
    code_column: str
    scope_values: tuple[str, ...]
    scope_column: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "table": self.table,
            "code_column": self.code_column,
            "scope_column": self.scope_column,
            "scope_values": list(self.scope_values),
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> SourceSpec:
        return cls(
            kind=SourceKind(obj["kind"]),
            table=str(obj["table"]),
            code_column=str(obj["code_column"]),
            scope_column=None if obj.get("scope_column") is None else str(obj["scope_column"]),
            scope_values=tuple(str(v) for v in obj["scope_values"]),
        )


def _identifier(value: str) -> str:
    """Validate a SQL identifier at the system boundary (specs come from JSON)."""
    if not _IDENTIFIER.match(value):
        raise ValueError(f"invalid SQL identifier: {value!r}")
    return value


_COUNT_TEMPLATE = (
    "SELECT code, COUNT(*) AS row_count FROM ({inner}) "
    "WHERE code IS NOT NULL AND TRIM(code) <> '' "
    "GROUP BY 1 ORDER BY row_count DESC, code"
)


def scoped_enumeration_sql(spec: SourceSpec) -> str:
    """Render the distinct-code enumeration SQL for one declared scope."""
    if not spec.scope_values:
        raise ValueError("at least one scope value is required")
    code = _identifier(spec.code_column)
    table = _identifier(spec.table)
    if spec.kind is SourceKind.RAW_SAMPLES:
        inner = " UNION ALL ".join(
            f"SELECT {code} AS code FROM {_identifier(v)}.{table}" for v in spec.scope_values
        )
    else:
        if spec.scope_column is None:
            raise ValueError("staging specs require a scope_column")
        values = ", ".join(f"'{_identifier(v)}'" for v in spec.scope_values)
        inner = (
            f"SELECT {code} AS code FROM {table} "
            f"WHERE {_identifier(spec.scope_column)} IN ({values})"
        )
    return _COUNT_TEMPLATE.format(inner=inner)


def enumerate_scoped_codes(cursor: _Cursor, spec: SourceSpec) -> list[tuple[str, int]]:
    """Enumerate ``(code, row_count)`` over a declared scope, richest first."""
    cursor.execute(scoped_enumeration_sql(spec))
    return [(str(row[0]), int(row[1])) for row in cursor.fetchall()]


@dataclass(frozen=True)
class MainTypeReport:
    """Per-code source-side main types, plus the scopes that could not be read.

    Separated because a total failure and a total gap are different facts: the
    worksheet must not present "no main type exists" when what happened is that
    every study raised.
    """

    main_types: dict[str, str]
    failures: dict[str, str]


def source_main_types(
    cursor: _Cursor,
    spec: SourceSpec,
    *,
    main_type_column: str = "CANCER_TYPE",
) -> MainTypeReport:
    """One source-side descriptive value per code, over a DECLARED scope.

    ``sample.CANCER_TYPE`` IS OncoTree's ``mainType`` and is source-side data, so
    it carries none of the target-vocabulary anchoring D4 rules out. Rendered per
    :class:`SourceKind` through the same validated identifiers as every other SQL
    path here — interpolating ``scope_values`` into a ``FROM`` clause worked only
    while a staging scope's study values happened to also be schema names.
    """
    code = _identifier(spec.code_column)
    main_type = _identifier(main_type_column)
    main_types: dict[str, str] = {}
    failures: dict[str, str] = {}
    for value in spec.scope_values:
        sql = _main_type_sql(spec, value, code=code, main_type=main_type)
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
        except Exception as exc:  # noqa: BLE001 — collected and reported, never dropped
            failures[value] = f"{type(exc).__name__}: {exc}"
            continue
        main_types.update({str(c): str(v) for c, v in rows})
    return MainTypeReport(main_types=main_types, failures=failures)


def _main_type_sql(spec: SourceSpec, value: str, *, code: str, main_type: str) -> str:
    select = f"SELECT {code}, ANY_VALUE({main_type})"
    if spec.kind is SourceKind.RAW_SAMPLES:
        source = f"{_identifier(value)}.{_identifier(spec.table)}"
        predicate = f"{main_type} IS NOT NULL"
    else:
        if spec.scope_column is None:
            raise ValueError("staging specs require a scope_column")
        source = _identifier(spec.table)
        predicate = (
            f"{_identifier(spec.scope_column)} = '{_identifier(value)}' "
            f"AND {main_type} IS NOT NULL"
        )
    return f"{select} FROM {source} WHERE {predicate} GROUP BY 1"


def raw_samples_view(
    spec: SourceSpec,
    *,
    table: str = "sample",
    code_column: str = "ONCOTREE_CODE",
) -> SourceSpec:
    """The same declared scope addressed as raw ``sample`` tables.

    G-01's two shapes describe the same gold set, so a staging scope's study
    values are also the raw study schemas. Source-side context (main type) only
    exists on the raw side, and stating that conversion here makes the assumption
    explicit and testable instead of hiding it inside a SQL string.
    """
    if spec.kind is SourceKind.RAW_SAMPLES:
        return spec
    from dataclasses import replace

    return replace(
        spec,
        kind=SourceKind.RAW_SAMPLES,
        table=table,
        code_column=code_column,
        scope_column=None,
    )


def discover_oncotree_schemas(cursor: _Cursor) -> list[str]:
    """Propose cbioportal_* schemas exposing ``sample.ONCOTREE_CODE``.

    Refresh input only — a *proposal* for a human to declare, never a scope.
    """
    cursor.execute(
        "SELECT DISTINCT table_schema FROM information_schema.columns "
        "WHERE column_name = 'ONCOTREE_CODE' AND table_name = 'sample' "
        "ORDER BY table_schema"
    )
    return [row[0] for row in cursor.fetchall()]
