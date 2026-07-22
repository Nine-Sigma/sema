"""US-012: mapping eval report — store reader + report assembly.

Runs the frozen §1.5(f) metric module (US-002) end-to-end over the resolved
decisions in the value-mapping store (US-005), graded against the gold set, and
applies the acceptance gate (:mod:`sema.eval.mapping_report_utils`). It READS
the store (US-006 is the sole writer) and READS the gold set — it certifies
nothing on Sema's own labels.

The metric math is NOT re-derived here: :func:`build_mapping_report` calls
:func:`sema.eval.mapping_goldset.score`, which owns the frozen confusion matrix.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from sema.eval.mapping_goldset import GoldSet, load_gold_set, score
from sema.eval.mapping_goldset_utils import Decision
from sema.eval.mapping_report_utils import (
    MappingReport,
    decision_from_value_mapping,
    evaluate_acceptance,
)
from sema.resolve.value_mapping_store import ValueMappingStore

__all__ = [
    "build_mapping_report",
    "decisions_from_store",
    "report_from_store",
]


def decisions_from_store(
    store: ValueMappingStore,
    *,
    source_vocabulary: str | None = None,
    target_property_ref: str | None = None,
    resolver_policy_ref: str | None = None,
    vocab_release: str | None = None,
) -> list[Decision]:
    """Read the store's resolved rows as scoring decisions (optionally scoped)."""
    decisions: list[Decision] = []
    for mapping in store.read_all():
        if source_vocabulary is not None and mapping.source_vocabulary != source_vocabulary:
            continue
        if target_property_ref is not None and mapping.target_property_ref != target_property_ref:
            continue
        if resolver_policy_ref is not None and mapping.resolver_policy_ref != resolver_policy_ref:
            continue
        if vocab_release is not None and mapping.vocab_release != vocab_release:
            continue
        decisions.append(decision_from_value_mapping(mapping))
    return decisions


def build_mapping_report(
    gold: GoldSet,
    decisions: Iterable[Decision],
) -> MappingReport:
    """Score decisions against the gold set and apply the acceptance gate."""
    score_report = score(gold.rows, decisions)
    coverage = gold.coverage_fraction()
    verdict, reason = evaluate_acceptance(score_report.distinct_code, coverage)
    return MappingReport(
        score=score_report,
        coverage_fraction=coverage,
        labelled_count=gold.labelled_count,
        total_codes=len(gold.rows),
        verdict=verdict,
        verdict_reason=reason,
        unlabelled_codes=tuple(gold.unlabelled_codes()),
    )


def report_from_store(
    store: ValueMappingStore,
    gold_path: str | Path,
    *,
    source_vocabulary: str | None = None,
    target_property_ref: str | None = None,
    resolver_policy_ref: str | None = None,
    vocab_release: str | None = None,
) -> MappingReport:
    """Convenience: read the gold set + scoped store decisions and report."""
    gold = GoldSet(load_gold_set(gold_path))
    decisions = decisions_from_store(
        store,
        source_vocabulary=source_vocabulary,
        target_property_ref=target_property_ref,
        resolver_policy_ref=resolver_policy_ref,
        vocab_release=vocab_release,
    )
    return build_mapping_report(gold, decisions)
