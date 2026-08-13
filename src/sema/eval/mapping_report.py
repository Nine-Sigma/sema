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

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sema.eval.mapping_goldset import (
    GoldSet,
    acceptance_eligible,
    load_gold_set,
    score,
)
from sema.eval.mapping_goldset_utils import Decision, GoldLabel
from sema.eval.adjudication import adjudication_qualifiers
from sema.eval.mapping_report_utils import (
    MappingReport,
    decision_from_value_mapping,
    evaluate_acceptance,
    tail_sample,
)
from sema.eval.mapping_run import EvaluationSubject
from sema.resolve.value_mapping_store import ValueMappingStore
from sema.resolve.value_mapping_store_utils import ValueMapping

if TYPE_CHECKING:  # importing the loader here would cycle back through eval
    from sema.eval.goldset_snapshot import GoldSetSnapshot

__all__ = [
    "GradingContext",
    "GradingReleaseError",
    "build_mapping_report",
    "decisions_from_store",
    "mappings_for_subject",
    "report_for_snapshot",
    "report_from_store",
]


class GradingReleaseError(ValueError):
    """The graded subject is not keyed to the release the gold set pins.

    A ``gold_concept_id`` is a bare integer, meaningless without the vocabulary
    release that minted it, so grading across releases reports vocabulary churn
    as resolver error. The pin lives here rather than in one CLI because every
    caller that grades must honour it.
    """


@dataclass(frozen=True)
class GradingContext:
    """Everything a snapshot declares about how its labels must be graded.

    The strata a report can see are a property of the DECLARATION, not of the
    rows: ``challenge_codes`` and ``tail_universe`` come from the header and the
    universe manifest, neither of which survives a bare JSONL read. Passing this
    one object is what keeps ``sema fit --strict`` and ``sema eval goldset
    mapping-report`` on the same grading path.
    """

    gold: GoldSet
    snapshot_version: str = ""
    challenge_codes: tuple[str, ...] = ()
    tail_universe: tuple[str, ...] | None = None
    vocab_release: str = ""

    @classmethod
    def from_snapshot(cls, snapshot: GoldSetSnapshot) -> GradingContext:
        frozen = {*snapshot.header.tier_codes, *snapshot.header.challenge_codes}
        return cls(
            gold=GoldSet(snapshot.rows),
            snapshot_version=snapshot.header.snapshot_version,
            challenge_codes=snapshot.header.challenge_codes,
            tail_universe=tuple(
                e.code for e in snapshot.universe if e.code not in frozen
            ),
            vocab_release=snapshot.header.vocab_release,
        )

    @classmethod
    def empty(cls) -> GradingContext:
        """Grade against no oracle at all — no labels, so no stratum and no pin."""
        return cls(gold=GoldSet(rows=[]))


def report_for_snapshot(
    context: GradingContext,
    decisions: Iterable[Decision],
    *,
    graded_release: str,
) -> MappingReport:
    """The single graded entry point: pin the release, then score every stratum.

    ``graded_release`` is read from the decisions' own store rows rather than
    from a caller's intent, so the pin is evidence about what was graded.
    """
    if context.vocab_release and graded_release != context.vocab_release:
        raise GradingReleaseError(
            f"the graded subject is pinned to {graded_release or '(unstated)'} but "
            f"the gold set {context.snapshot_version or '(unversioned)'} keys its "
            f"concept ids to {context.vocab_release}. A gold_concept_id is "
            "meaningless without the release that minted it, so grading across "
            "releases would report vocabulary churn as resolver error. "
            "Re-snapshot the gold set, or grade the release it pins."
        )
    return build_mapping_report(
        context.gold,
        decisions,
        snapshot_version=context.snapshot_version,
        challenge_codes=context.challenge_codes,
        tail_universe=context.tail_universe,
    )


def mappings_for_subject(
    store: ValueMappingStore,
    subject: EvaluationSubject,
) -> list[ValueMapping]:
    """Read exactly the store rows that belong to one evaluation subject.

    Returns the store rows rather than projected decisions, so a caller that also
    needs the store-side provenance the projection drops (``run_id``) reads the
    store ONCE — grading one decision set and reporting on another would make the
    immutable run artifact disagree with its own report.
    """
    return [
        mapping
        for mapping in store.read_all()
        if mapping.source_vocabulary == subject.source_vocabulary
        and mapping.target_property_ref == subject.target_property_ref
        and mapping.resolver_policy_ref == subject.resolver_policy_ref
        and mapping.vocab_release == subject.vocab_release
    ]


def decisions_from_store(
    store: ValueMappingStore,
    subject: EvaluationSubject,
) -> list[Decision]:
    """One evaluation subject's store rows, projected onto scoring decisions."""
    return [decision_from_value_mapping(m) for m in mappings_for_subject(store, subject)]


def build_mapping_report(
    gold: GoldSet,
    decisions: Iterable[Decision],
    *,
    snapshot_version: str = "",
    challenge_codes: Sequence[str] | None = None,
    tail_universe: Sequence[str] | None = None,
) -> MappingReport:
    """Score decisions against the gold set and apply the acceptance gate."""
    score_report = score(gold.rows, decisions)
    coverage = gold.coverage_fraction()
    by_code = gold.by_code()
    ungraded = tuple(
        code
        for code in score_report.labelled_without_decision
        if acceptance_eligible(by_code[code])
    )
    verdict, reason = evaluate_acceptance(
        score_report.distinct_code, coverage, ungraded_codes=ungraded
    )
    sample = tail_sample(gold.rows, snapshot_version, tail_universe=tail_universe)
    return MappingReport(
        score=score_report,
        coverage_fraction=coverage,
        labelled_count=gold.labelled_count,
        total_codes=gold.total_eligible_codes,
        verdict=verdict,
        verdict_reason=reason,
        unlabelled_codes=tuple(gold.unlabelled_codes()),
        snapshot_version=snapshot_version,
        qualifiers=adjudication_qualifiers(gold.rows, challenge_codes),
        tail_sample_codes=sample,
        tail_sample_labelled=sum(
            1
            for c in sample
            if c in by_code and by_code[c].gold_label is not GoldLabel.UNLABELLED
        ),
        tail_sample_unlabellable=sum(1 for c in sample if c not in by_code),
        ungraded_codes=ungraded,
    )


def report_from_store(
    store: ValueMappingStore,
    gold_path: str | Path,
    *,
    subject: EvaluationSubject,
    snapshot_version: str = "",
    challenge_codes: Sequence[str] | None = None,
) -> MappingReport:
    """Read the gold set + the subject's store decisions and report."""
    gold = GoldSet(load_gold_set(gold_path))
    return build_mapping_report(
        gold,
        decisions_from_store(store, subject),
        snapshot_version=snapshot_version,
        challenge_codes=challenge_codes,
    )
