"""S1-02 — deterministic identity resolver.

Maps source rows to canonical identities through the :mod:`identity_registry`:
``(source_namespace, source_entity_key) -> entity_id``. A row whose entity key is
missing/blank is NOT given a synthetic identity and NOT silently dropped — it is
routed to a typed review disposition (D5). The registry is the sole identity
writer; this resolver is its only caller in the fit chain.

The module is DOMAIN-GENERIC (D6/R29): the missing-key reason code is injected
by the policy layer (the OMOP ``MISSING_PERSON_KEY`` binding lives in
:mod:`sema.resolve.policies.omop`), so nothing here names ``person``/OMOP.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from sema.resolve.identity_registry import IdentityRegistry


@dataclass(frozen=True)
class IdentitySourceRow:
    """One source row's identity-relevant fields.

    ``source_entity_key`` may be ``None``/blank (the missing-key case);
    ``source_row_ref`` is the stable per-row identity (e.g. the source PK) used
    both to trace review dispositions and to key the eventual row surrogate PK.
    """

    source_namespace: str
    source_entity_key: str | None
    source_row_ref: str


@dataclass(frozen=True)
class ResolvedIdentityRow:
    """A source row resolved to a canonical ``entity_id``."""

    source_row_ref: str
    source_namespace: str
    source_entity_key: str
    entity_id: int


@dataclass(frozen=True)
class MissingKeyDisposition:
    """A source row routed to review because it carries no usable entity key."""

    source_row_ref: str
    source_namespace: str
    reason: str


@dataclass(frozen=True)
class IdentityResolution:
    """The partitioned outcome of resolving a batch of source rows."""

    resolved: list[ResolvedIdentityRow]
    review: list[MissingKeyDisposition]

    @property
    def resolved_count(self) -> int:
        return len(self.resolved)

    @property
    def review_count(self) -> int:
        return len(self.review)


def _normalized_key(key: str | None) -> str | None:
    """Return the trimmed key, or ``None`` when it is missing/whitespace-only."""
    if key is None:
        return None
    trimmed = key.strip()
    return trimmed or None


class DeterministicIdentityResolver:
    """Resolve source rows to canonical identities; route missing keys to review."""

    def __init__(
        self, registry: IdentityRegistry, *, missing_key_reason: str
    ) -> None:
        if not missing_key_reason:
            raise ValueError("missing_key_reason must be a non-empty typed code")
        self._registry = registry
        self._reason = missing_key_reason

    @property
    def registry(self) -> IdentityRegistry:
        return self._registry

    def resolve(
        self, rows: Iterable[IdentitySourceRow], *, run_id: str
    ) -> IdentityResolution:
        review: list[MissingKeyDisposition] = []
        resolvable: list[tuple[IdentitySourceRow, str]] = []
        for row in rows:
            key = _normalized_key(row.source_entity_key)
            if key is None:
                review.append(
                    MissingKeyDisposition(
                        source_row_ref=row.source_row_ref,
                        source_namespace=row.source_namespace,
                        reason=self._reason,
                    )
                )
            else:
                resolvable.append((row, key))

        pairs = [(row.source_namespace, key) for row, key in resolvable]
        assignments = (
            self._registry.get_or_create(pairs, run_id=run_id) if pairs else {}
        )
        resolved = [
            ResolvedIdentityRow(
                source_row_ref=row.source_row_ref,
                source_namespace=row.source_namespace,
                source_entity_key=key,
                entity_id=assignments[(row.source_namespace, key)].entity_id,
            )
            for row, key in resolvable
        ]
        return IdentityResolution(resolved=resolved, review=review)
