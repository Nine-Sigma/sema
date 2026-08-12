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
    "SourceKind",
    "SourceSpec",
    "discover_oncotree_schemas",
    "enumerate_scoped_codes",
    "scoped_enumeration_sql",
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
