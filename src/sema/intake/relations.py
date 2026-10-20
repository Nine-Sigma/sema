"""Resolve a spec relation to a read query against the warehouse or staging.

The importer never knows where a relation lives: it asks the provider for the
query function of each ``RelationRef``. Statements are built by the caller from
spec identifiers only; the provider runs them unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from sema.binding.spec_sections import RelationRef
from sema.ingest.duckdb_staging import Staging

QueryFn = Callable[[str], list[tuple[Any, ...]]]


class MissingProviderError(ValueError):
    """A spec names a provider the caller did not supply."""


@dataclass(frozen=True)
class RelationProvider:
    warehouse: QueryFn | None
    staging: Staging | None

    def query(self, ref: RelationRef) -> QueryFn:
        if ref.provider == "staging":
            return self._staging_query(ref)
        if self.warehouse is None:
            raise _missing(ref)
        return self.warehouse

    def require(self, refs: Iterable[RelationRef]) -> None:
        for ref in refs:
            if not self._supplies(ref):
                raise _missing(ref)

    def _supplies(self, ref: RelationRef) -> bool:
        supplied = self.staging if ref.provider == "staging" else self.warehouse
        return supplied is not None

    def _staging_query(self, ref: RelationRef) -> QueryFn:
        staging = self.staging
        if staging is None:
            raise _missing(ref)

        def run(sql: str) -> list[tuple[Any, ...]]:
            rows: list[tuple[Any, ...]] = staging.execute(sql).fetchall()
            return rows

        return run


def _missing(ref: RelationRef) -> MissingProviderError:
    option = " (pass --staging <duckdb path>)" if ref.provider == "staging" else ""
    return MissingProviderError(
        f"the {ref.provider} provider was not supplied and relation {ref.name} "
        f"needs it{option}"
    )
