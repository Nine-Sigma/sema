"""Physical-catalog port (D1) for the target-retrieval projection.

The retrieval graph binds a target semantic model to *physical* tables and
columns. Their metadata is read from the live warehouse catalog at load time
(the authored manifest can drift from what was actually written), so the read
sits behind this port: a live implementation reads `information_schema`; a
static implementation backs deterministic tests.

Domain-neutral by construction — it knows only catalog/schema/table/column,
never any target vocabulary or entity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class CatalogDriftError(RuntimeError):
    """A manifest property has no matching physical column in the catalog.

    Raised loudly rather than silently skipped: a drifted binding would
    produce a target graph that points at columns which do not exist.
    """


@dataclass(frozen=True)
class PhysicalColumn:
    name: str
    data_type: str
    nullable: bool


class PhysicalCatalogSource(Protocol):
    """Reads the physical columns of one table from a warehouse catalog."""

    def columns(
        self, *, catalog: str, schema: str, table: str
    ) -> list[PhysicalColumn]: ...


class StaticPhysicalCatalogSource:
    """In-memory `PhysicalCatalogSource` for tests and fixtures.

    Keyed by ``(catalog, schema, table)``; an unknown table yields ``[]`` so
    the loader's drift guard fires exactly as it would against a live catalog
    missing that table.
    """

    def __init__(
        self, tables: dict[tuple[str, str, str], list[PhysicalColumn]]
    ) -> None:
        self._tables = tables

    def columns(
        self, *, catalog: str, schema: str, table: str
    ) -> list[PhysicalColumn]:
        return list(self._tables.get((catalog, schema, table), []))


__all__ = [
    "CatalogDriftError",
    "PhysicalCatalogSource",
    "PhysicalColumn",
    "StaticPhysicalCatalogSource",
]
