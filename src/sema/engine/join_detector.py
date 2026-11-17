"""Domain-agnostic FK / join detector.

Enumerates intra-schema FK candidates from column metadata, verifies
each via tiered evidence, and emits `FK_TO` assertions tagged with
`source_schema`. Tier 1 checks a bounded distinct sample of the
referencing column against the whole referenced column, one statement
per candidate; the detector NEVER issues unbounded
`COUNT(... NOT IN ...)` referential-integrity scans. When the check is
inconclusive, the detector downgrades confidence rather than escalating
cost. A referenced column the profile shows is not unique, or one the
check refutes, yields no assertion; one with no profile stays at Tier 3.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from sema.engine.join_detector_utils import (
    COVERAGE_THRESHOLD as COVERAGE_THRESHOLD,
    FKCandidate,
    enumerate_candidates_from_metadata,
    verify_cardinality,
)
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)
from sema.models.extraction import ExtractedColumn

ColumnKey = tuple[str, str, str]
Profiles = dict[ColumnKey, tuple[int, int]]
SamplerFn = Callable[[ColumnKey], set[str] | None]
ContainmentFn = Callable[[ColumnKey, frozenset[str]], int | None]
DEFAULT_SAMPLE_CAP = 500
DEFAULT_MATERIALIZE_THRESHOLD = 0.80
TIER_1 = 0.95
TIER_2 = 0.80
TIER_3 = 0.70


@dataclass(frozen=True)
class FKAssertion:
    """Detector output: candidate + tier + confidence + provenance."""
    candidate: FKCandidate
    confidence: float
    tier: int
    source_schema: str


def _column_key(table: str, column: str, schema: str) -> ColumnKey:
    return (schema, table, column)


def _referenced_not_unique(
    candidate: FKCandidate, profiles: Profiles | None,
) -> bool:
    stats = (profiles or {}).get(
        _column_key(candidate.pk_table, candidate.pk_column, candidate.schema_name),
    )
    return stats is not None and stats[0] != stats[1]


@dataclass
class JoinDetector:
    """FK / join candidate detector.

    `sample_cap` bounds the referencing values sent in one containment
    statement. Without a containment answer the detector downgrades to
    Tier 2 if cardinality metadata supports the FK relation, else
    Tier 3. The detector NEVER scales up to unbounded RI queries.
    """
    sample_cap: int = DEFAULT_SAMPLE_CAP
    materialization_threshold: float = DEFAULT_MATERIALIZE_THRESHOLD

    def detect(
        self,
        *,
        columns: list[ExtractedColumn],
        source_schema: str,
        profiles: Profiles | None = None,
        sampler: SamplerFn | None = None,
        containment: ContainmentFn | None = None,
    ) -> list[FKAssertion]:
        out: list[FKAssertion] = []
        for candidate in enumerate_candidates_from_metadata(columns):
            if _referenced_not_unique(candidate, profiles):
                continue
            fk = self._classify(
                candidate, source_schema, profiles, sampler, containment,
            )
            if fk is not None:
                out.append(fk)
        return out

    def should_materialize(self, fk: FKAssertion) -> bool:
        return fk.confidence >= self.materialization_threshold

    def _classify(
        self,
        candidate: FKCandidate,
        source_schema: str,
        profiles: Profiles | None,
        sampler: SamplerFn | None,
        containment: ContainmentFn | None,
    ) -> FKAssertion | None:
        pk_key = _column_key(
            candidate.pk_table, candidate.pk_column, candidate.schema_name,
        )
        fk_key = _column_key(
            candidate.fk_table, candidate.fk_column, candidate.schema_name,
        )
        if pk_key not in (profiles or {}):
            return FKAssertion(candidate, TIER_3, 3, source_schema)
        coverage = self._coverage(pk_key, fk_key, sampler, containment)
        if coverage is not None:
            if coverage < COVERAGE_THRESHOLD:
                return None
            return FKAssertion(candidate, TIER_1, 1, source_schema)

        tier2 = self._try_tier_2(pk_key, fk_key, profiles)
        if tier2 is not None:
            return FKAssertion(candidate, tier2, 2, source_schema)

        return FKAssertion(candidate, TIER_3, 3, source_schema)

    def _coverage(
        self,
        pk_key: ColumnKey,
        fk_key: ColumnKey,
        sampler: SamplerFn | None,
        containment: ContainmentFn | None,
    ) -> float | None:
        """Share of the referencing sample found in the whole referenced column."""
        if sampler is None or containment is None:
            return None
        try:
            fk_sample = sampler(fk_key)
        except Exception:
            return None
        if not fk_sample:
            return None
        values = frozenset(sorted(fk_sample)[: self.sample_cap])
        matched = containment(pk_key, values)
        if matched is None:
            return None
        return matched / len(values)

    def _try_tier_2(
        self,
        pk_key: ColumnKey,
        fk_key: ColumnKey,
        profiles: Profiles | None,
    ) -> float | None:
        if not profiles:
            return None
        pk_stats = profiles.get(pk_key)
        fk_stats = profiles.get(fk_key)
        if pk_stats is None or fk_stats is None:
            return None
        pk_distinct, pk_rows = pk_stats
        fk_distinct, _ = fk_stats
        if not verify_cardinality(pk_distinct, pk_rows, fk_distinct):
            return None
        return TIER_2


def to_fk_assertion(
    fk: FKAssertion, run_id: str,
) -> Assertion:
    return Assertion(
        id=str(uuid.uuid4()),
        subject_ref=fk.candidate.fk_ref,
        predicate=AssertionPredicate.FK_TO,
        payload={
            "pk_table": fk.candidate.pk_table,
            "pk_column": fk.candidate.pk_column,
            "fk_table": fk.candidate.fk_table,
            "fk_column": fk.candidate.fk_column,
            "tier": fk.tier,
        },
        object_ref=fk.candidate.pk_ref,
        source="fk_detector",
        confidence=fk.confidence,
        status=AssertionStatus.AUTO,
        run_id=run_id,
        observed_at=datetime.now(timezone.utc),
        source_schema=fk.source_schema,
    )
