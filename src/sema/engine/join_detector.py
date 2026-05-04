"""Domain-agnostic FK / join detector.

Enumerates intra-schema FK candidates from column metadata, verifies
each via tiered evidence, and emits `FK_TO` assertions tagged with
`source_schema`. Verification uses bounded distinct-value samples
when available; the detector NEVER issues unbounded
`COUNT(... NOT IN ...)` referential-integrity scans — when sample-based
verification is inconclusive, the detector downgrades confidence
rather than escalating cost.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from sema.engine.join_detector_utils import (
    FKCandidate,
    coverage_ratio,
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
SamplerFn = Callable[[ColumnKey], set[str] | None]
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


@dataclass
class JoinDetector:
    """FK / join candidate detector.

    `sample_cap` bounds detector-owned distinct-value samples. When FK
    distinct cardinality exceeds this cap the detector downgrades to
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
        profiles: dict[ColumnKey, tuple[int, int]] | None = None,
        samples: dict[ColumnKey, set[str]] | None = None,
        sampler: SamplerFn | None = None,
    ) -> list[FKAssertion]:
        candidates = enumerate_candidates_from_metadata(columns)
        return [
            self._classify(c, source_schema, profiles, samples, sampler)
            for c in candidates
        ]

    def should_materialize(self, fk: FKAssertion) -> bool:
        return fk.confidence >= self.materialization_threshold

    def _classify(
        self,
        candidate: FKCandidate,
        source_schema: str,
        profiles: dict[ColumnKey, tuple[int, int]] | None,
        samples: dict[ColumnKey, set[str]] | None,
        sampler: SamplerFn | None,
    ) -> FKAssertion:
        pk_key = _column_key(
            candidate.pk_table, candidate.pk_column, candidate.schema_name,
        )
        fk_key = _column_key(
            candidate.fk_table, candidate.fk_column, candidate.schema_name,
        )

        pk_sample, fk_sample, sample_origin = self._collect_samples(
            pk_key, fk_key, samples, sampler,
        )
        if pk_sample is not None and fk_sample is not None:
            tier1 = self._try_tier_1(pk_sample, fk_sample)
            if tier1 is not None:
                return FKAssertion(
                    candidate, tier1, 1, source_schema,
                )

        tier2 = self._try_tier_2(pk_key, fk_key, profiles)
        if tier2 is not None:
            return FKAssertion(candidate, tier2, 2, source_schema)

        return FKAssertion(candidate, TIER_3, 3, source_schema)

    def _collect_samples(
        self,
        pk_key: ColumnKey,
        fk_key: ColumnKey,
        samples: dict[ColumnKey, set[str]] | None,
        sampler: SamplerFn | None,
    ) -> tuple[set[str] | None, set[str] | None, str]:
        if samples and pk_key in samples and fk_key in samples:
            return samples[pk_key], samples[fk_key], "profiler"
        if sampler is None:
            return None, None, "missing"
        try:
            pk = sampler(pk_key)
            fk = sampler(fk_key)
        except Exception:
            return None, None, "warehouse_error"
        if pk is None or fk is None:
            return None, None, "missing"
        return pk, fk, "detector"

    def _try_tier_1(
        self, pk_sample: set[str], fk_sample: set[str],
    ) -> float | None:
        """Tier 1 sample subset test.

        When FK distinct cardinality has hit the sample cap the subset
        relation is unprovable — return None so the caller falls
        through to Tier 2 / Tier 3 without escalating to RI scans.
        """
        if len(fk_sample) >= self.sample_cap:
            return None
        if not fk_sample:
            return None
        if coverage_ratio(fk_sample, pk_sample) < 0.80:
            return None
        return TIER_1

    def _try_tier_2(
        self,
        pk_key: ColumnKey,
        fk_key: ColumnKey,
        profiles: dict[ColumnKey, tuple[int, int]] | None,
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
    cat = fk.candidate.catalog or ""
    schema = fk.candidate.schema_name
    fk_ref = (
        f"databricks://workspace/{cat}/{schema}/"
        f"{fk.candidate.fk_table}/{fk.candidate.fk_column}"
    )
    pk_ref = (
        f"databricks://workspace/{cat}/{schema}/"
        f"{fk.candidate.pk_table}/{fk.candidate.pk_column}"
    )
    return Assertion(
        id=str(uuid.uuid4()),
        subject_ref=fk_ref,
        predicate=AssertionPredicate.FK_TO,
        payload={
            "pk_table": fk.candidate.pk_table,
            "pk_column": fk.candidate.pk_column,
            "fk_table": fk.candidate.fk_table,
            "fk_column": fk.candidate.fk_column,
            "tier": fk.tier,
        },
        object_ref=pk_ref,
        source="fk_detector",
        confidence=fk.confidence,
        status=AssertionStatus.AUTO,
        run_id=run_id,
        observed_at=datetime.now(timezone.utc),
        source_schema=fk.source_schema,
    )
