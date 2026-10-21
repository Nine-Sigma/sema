"""The Land contract: every loader returns one ``LandResult``.

A result names the relations a loader wrote, a provenance record per relation
(design section 5.1.1), and the source identity. ``schema_snapshot`` is derived
from the relations' names and columns only, so two landings of one source
compare equal although ``landed_at`` differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from sema.ingest.duckdb_staging_utils import escape_sql_literal

LOCATOR_COLUMN = "_sema_locator"
LOCATOR_ROW_SEPARATOR = "!"
SHEET_SEPARATOR = "__"

Medium = Literal["tabular", "semsql", "dictionary", "athena", "database", "warehouse"]
LocatorKind = Literal["row", "relation"]
Columns = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Provenance:
    """``locator`` is set only for ``locator_kind: relation``; row loaders use a column."""

    source_uri: str
    loader: str
    loader_version: str
    landed_at: str
    policy_class: str
    locator_kind: LocatorKind
    locator: str | None = None


@dataclass(frozen=True)
class LandedRelation:
    schema: str
    name: str
    columns: Columns
    row_count: int
    provenance: Provenance
    comments: tuple[tuple[str, str], ...] = ()
    header_row: int | None = None
    empty_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class LandResult:
    medium: Medium
    source_identity: tuple[tuple[str, str], ...]
    relations: tuple[LandedRelation, ...]
    policy_class: str
    headerless: tuple[str, ...] = ()

    @property
    def schema_snapshot(self) -> tuple[tuple[str, Columns], ...]:
        return tuple((r.name, r.columns) for r in self.relations)

    def relation(self, name: str) -> LandedRelation:
        return next(r for r in self.relations if r.name == name)


def locator_order_sql(quoted_locator: str) -> str:
    """Landed row order in staging SQL: the number after the last separator, then the text.

    It orders as ``locator_row`` sorts, so a numeric row part orders numerically (row 2
    before row 10) and a locator without one sorts last.
    """
    separator = escape_sql_literal(LOCATOR_ROW_SEPARATOR)
    number = f"TRY_CAST(string_split({quoted_locator}, '{separator}')[-1] AS BIGINT)"
    return f"{number} NULLS LAST, {quoted_locator}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
