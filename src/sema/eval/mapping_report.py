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
from sema.eval.mapping_goldset_utils import Decision, GoldLabel
from sema.eval.mapping_report_utils import (
    MappingReport,
    adjudication_qualifiers,
    decision_from_value_mapping,
    evaluate_acceptance,
    tail_sample,
)
from sema.eval.mapping_run import EvaluationSubject
from sema.resolve.value_mapping_store import ValueMappingStore

__all__ = [
    "build_mapping_report",
    "decisions_from_store",
    "report_from_store",
]


def decisions_from_store(
    store: ValueMappingStore,
    subject: EvaluationSubject,
) -> list[Decision]:
    """Read exactly the store rows that belong to one evaluation subject."""
    return [
        decision_from_value_mapping(mapping)
        for mapping in store.read_all()
        if mapping.source_vocabulary == subject.source_vocabulary
        and mapping.target_property_ref == subject.target_property_ref
        and mapping.resolver_policy_ref == subject.resolver_policy_ref
        and mapping.vocab_release == subject.vocab_release
    ]


def build_mapping_report(
    gold: GoldSet,
    decisions: Iterable[Decision],
    *,
    snapshot_version: str = "",
) -> MappingReport:
    """Score decisions against the gold set and apply the acceptance gate."""
    score_report = score(gold.rows, decisions)
    coverage = gold.coverage_fraction()
    verdict, reason = evaluate_acceptance(score_report.distinct_code, coverage)
    sample = tail_sample(gold.rows, snapshot_version)
    by_code = gold.by_code()
    return MappingReport(
        score=score_report,
        coverage_fraction=coverage,
        labelled_count=gold.labelled_count,
        total_codes=gold.total_eligible_codes,
        verdict=verdict,
        verdict_reason=reason,
        unlabelled_codes=tuple(gold.unlabelled_codes()),
        snapshot_version=snapshot_version,
        qualifiers=adjudication_qualifiers(gold.rows),
        tail_sample_codes=sample,
        tail_sample_labelled=sum(
            1 for c in sample if by_code[c].gold_label is not GoldLabel.UNLABELLED
        ),
    )


def report_from_store(
    store: ValueMappingStore,
    gold_path: str | Path,
    *,
    subject: EvaluationSubject,
    snapshot_version: str = "",
) -> MappingReport:
    """Read the gold set + the subject's store decisions and report."""
    gold = GoldSet(load_gold_set(gold_path))
    return build_mapping_report(
        gold, decisions_from_store(store, subject), snapshot_version=snapshot_version
    )
