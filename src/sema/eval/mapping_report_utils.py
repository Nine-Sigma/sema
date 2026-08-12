"""US-012: mapping eval report — acceptance logic, caveat, and shapes.

Lives under ``eval/`` (R29-allowlisted) so it may read the resolved
value-mapping store (US-005) — whose ``concept_id`` field name is on the R29
denylist — and reconcile against the gold set without tripping the coupling
guard. The §1.5(f) metric math is NOT re-derived here: it is imported from the
frozen US-002 module (:mod:`sema.eval.mapping_goldset_utils`).

Acceptance is gated, never self-certified. The report is ``accepted`` ONLY at
100% human-labelled gold coverage AND with ``mapped_precision`` >= 95% and
``auto_resolution_rate`` >= 70%. Below 100% coverage it is ``provisional — not
accepted`` (the human-label gate from US-002 is incomplete); at full coverage
below either threshold it is ``running, not accepted``. ``no_map_accuracy`` is
reported but is deliberately NOT a gating threshold.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sema.eval.mapping_goldset import GoldSetReport
from sema.eval.mapping_goldset_utils import (
    ConfusionMatrix,
    Decision,
    GoldLabel,
    GoldRow,
    ResolutionStatus,
    TierState,
)
from sema.resolve.value_mapping_store_utils import ValueMapping

# §1.5(f) acceptance thresholds (asserted live only at full gold coverage).
MIN_MAPPED_PRECISION = 0.95
MIN_AUTO_RESOLUTION_RATE = 0.70

# Slice 0's precision is a property of the deterministic walk, not the product.
STRUCTURAL_PRECISION_CAVEAT = (
    "Slice 0's near-100% precision is STRUCTURAL: it is produced by a "
    "deterministic exact-code walk (source code -> standardize -> domain gate), "
    "NOT by a learned or fuzzy matcher. It does NOT validate the product "
    "precision approach on the ambiguous tail (fuzzy names, multi-survivor "
    "ties, the disambiguation council)."
)


class AcceptanceVerdict(str, Enum):
    """Whether the run may be called 'accepted' — never self-certified.

    Acceptance is scope-qualified by name. A bare ``accepted`` over a head chosen
    *by row frequency* would overclaim: it says nothing about the distinct-code
    tail, and ``per_bucket`` only buckets rows that were labelled and scored, so
    it is not evidence about an unlabelled one either.
    """

    ACCEPTED = "accepted_for_frozen_frequency_head"
    PROVISIONAL_NOT_ACCEPTED = "provisional — not accepted"
    RUNNING_NOT_ACCEPTED = "running, not accepted"


# Second-review floor before a verdict may drop the ``unadjudicated`` qualifier:
# every labelled challenge code (a wrong gold NO_MAP scores fp_map directly
# against a correctly-mapped code) plus a sample of head labels.
MIN_SECOND_REVIEWED_HEAD_LABELS = 10


def adjudication_qualifiers(
    rows: Sequence[GoldRow],
    challenge_codes: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """``('unadjudicated',)`` unless the G-05 second-review floor was met.

    The floor is defined over the snapshot's DECLARATION, not over whatever has
    been labelled so far, because every row-derived shortcut silently weakens it:

    * ``challenge_codes`` must come from ``GoldSetHeader.challenge_codes``. A
      declared challenge code that also ranks inside the head (live case:
      ``IMMC``) resolves to ``IN_TIER``, so a ``tier_state`` check could never
      demand its review — and an entirely unlabelled stratum passed vacuously.
    * the head sample size is ``min(10, len(head))`` over the WHOLE head; taken
      over the labelled head it collapsed to "all of whatever you've labelled".
    * an out-of-tier label is not evidence of an adjudicated oracle.
    """
    declared = (
        frozenset(challenge_codes)
        if challenge_codes is not None
        else frozenset(r.oncotree_code for r in rows if r.tier_state is TierState.CHALLENGE)
    )
    head = [r for r in rows if r.tier_state is TierState.IN_TIER]
    labelled_head = [r for r in head if r.gold_label is not GoldLabel.UNLABELLED]
    if not labelled_head:
        return ("unadjudicated",)
    if codes_needing_second_review(rows, declared):
        return ("unadjudicated",)
    reviewed = sum(1 for r in labelled_head if r.second_reviewer)
    if reviewed < min(MIN_SECOND_REVIEWED_HEAD_LABELS, len(head)):
        return ("unadjudicated",)
    return ()


def codes_needing_second_review(
    rows: Sequence[GoldRow],
    challenge_codes: Sequence[str] | frozenset[str],
) -> list[str]:
    """Declared challenge codes not yet labelled AND second-reviewed.

    Unlabelled counts as needing review: a wrong gold ``NO_MAP`` scores
    ``fp_map`` straight against a correctly-mapped code, so this stratum is the
    one place a single bad label damages ``mapped_precision`` directly.
    """
    by_code = {r.oncotree_code: r for r in rows}
    return sorted(
        code
        for code in challenge_codes
        if (row := by_code.get(code)) is None
        or row.gold_label is GoldLabel.UNLABELLED
        or not row.second_reviewer
    )


def tail_sample(
    rows: Sequence[GoldRow],
    snapshot_version: str,
    size: int = 20,
    tail_universe: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """A deterministic, NEVER-gating sample of codes outside the frozen populations.

    The only stratum that speaks to "confidently wrong in the tail" — so it must be
    drawn from the universe manifest's tail when one is available. Drawing from the
    gold rows alone samples the codes that happened to be in a PRIOR scope, which is
    a biased slice of the tail rather than a sample of it.

    Derived from the frozen snapshot (version + manifest) so it is reproducible
    without becoming a second benchmark.
    """
    codes = list(tail_universe) if tail_universe is not None else [
        r.oncotree_code for r in rows if r.tier_state is TierState.OUT_OF_TIER
    ]
    ranked = sorted(
        codes, key=lambda c: hashlib.sha256(f"{snapshot_version}:{c}".encode()).hexdigest()
    )
    return tuple(sorted(ranked[:size]))


def decision_from_value_mapping(mapping: ValueMapping) -> Decision:
    """Project a §1.5(a) store row onto a scoring :class:`Decision`.

    The store's :class:`ResolutionStatus` mirrors the eval enum by value, so the
    conversion is value-preserving across the two (deliberately separate) enums.
    """
    return Decision(
        source_code=mapping.normalized_source_value,
        concept_id=mapping.concept_id,
        status=mapping.status,
        resolution_status=ResolutionStatus(mapping.resolution_status.value),
        no_map_reason=mapping.no_map_reason,
    )


def evaluate_acceptance(
    matrix: ConfusionMatrix | None,
    coverage_fraction: float,
    *,
    ungraded_codes: Sequence[str] = (),
) -> tuple[AcceptanceVerdict, str]:
    """Apply the §1.5(f) acceptance gate to one confusion matrix + coverage.

    ``ungraded_codes`` are acceptance-eligible codes that carry a human label but
    no decision in the graded subject. Labelling them is not the same as grading
    them: without this check a subject matching only part of the store (a
    mistyped ``resolver_policy_ref``, a release the store never held) reports
    100% coverage and a flawless matrix over whatever it happened to find.
    """
    if coverage_fraction < 1.0:
        return (
            AcceptanceVerdict.PROVISIONAL_NOT_ACCEPTED,
            f"labelled gold coverage {coverage_fraction:.1%} < 100% of the frozen "
            "tier (D1(x)); the US-002 human-label gate is incomplete",
        )
    if ungraded_codes:
        listed = ", ".join(sorted(ungraded_codes)[:10])
        return (
            AcceptanceVerdict.RUNNING_NOT_ACCEPTED,
            f"{len(ungraded_codes)} labelled code(s) in the frozen tier have no "
            f"decision in the graded subject ({listed}); the decision set does "
            "not cover the population the verdict would certify",
        )
    assert matrix is not None  # full coverage implies a scored matrix
    precision = matrix.mapped_precision
    auto = matrix.auto_resolution_rate
    if precision is None or auto is None:
        return (
            AcceptanceVerdict.RUNNING_NOT_ACCEPTED,
            "metrics undefined at full coverage "
            f"(precision={_pct(precision)}, auto_resolution={_pct(auto)})",
        )
    if precision >= MIN_MAPPED_PRECISION and auto >= MIN_AUTO_RESOLUTION_RATE:
        return (
            AcceptanceVerdict.ACCEPTED,
            f"mapped_precision {precision:.1%} >= {MIN_MAPPED_PRECISION:.0%} and "
            f"auto_resolution_rate {auto:.1%} >= {MIN_AUTO_RESOLUTION_RATE:.0%}",
        )
    return (
        AcceptanceVerdict.RUNNING_NOT_ACCEPTED,
        f"mapped_precision {precision:.1%} (>= {MIN_MAPPED_PRECISION:.0%}?) / "
        f"auto_resolution_rate {auto:.1%} (>= {MIN_AUTO_RESOLUTION_RATE:.0%}?) "
        "below threshold",
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


@dataclass(frozen=True)
class MappingReport:
    """The structured §1.5(f) report with the acceptance verdict + caveat."""

    score: GoldSetReport
    coverage_fraction: float
    labelled_count: int
    total_codes: int
    verdict: AcceptanceVerdict
    verdict_reason: str
    unlabelled_codes: tuple[str, ...] = ()
    snapshot_version: str = ""
    qualifiers: tuple[str, ...] = ()
    tail_sample_codes: tuple[str, ...] = ()
    tail_sample_labelled: int = 0
    tail_sample_unlabellable: int = 0
    ungraded_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "verdict_reason": self.verdict_reason,
            "qualifiers": list(self.qualifiers),
            "snapshot_version": self.snapshot_version,
            "coverage": {
                "labelled": self.labelled_count,
                "total": self.total_codes,
                "fraction": self.coverage_fraction,
                "unlabelled_codes": list(self.unlabelled_codes),
            },
            "graded": {
                "scored": self.score.scored_codes,
                "of": self.total_codes,
                "ungraded_codes": list(self.ungraded_codes),
            },
            "thresholds": {
                "mapped_precision": MIN_MAPPED_PRECISION,
                "auto_resolution_rate": MIN_AUTO_RESOLUTION_RATE,
            },
            "metrics": self.score.as_dict(),
            "strata": {
                "frozen_head": {"gating": True, "scored_codes": self.score.scored_codes},
                "challenge": {
                    "gating": False,
                    "scored_codes": self.score.challenge_scored_codes,
                },
                "random_tail": {
                    "gating": False,
                    "codes": list(self.tail_sample_codes),
                    "labelled": self.tail_sample_labelled,
                    "without_a_gold_row": self.tail_sample_unlabellable,
                    "scored_codes": self.score.tail_scored_codes,
                    "distinct_code": self.score.tail_distinct_code.as_dict(),
                    "note": (
                        "informational only — the frozen head is selected by row "
                        "frequency and says nothing about the distinct-code tail. "
                        "`without_a_gold_row` codes cannot be labelled until the "
                        "snapshot carries a row for them, so a zero labelled count "
                        "is not evidence that the tail is clean"
                    ),
                },
            },
            "structural_precision_caveat": STRUCTURAL_PRECISION_CAVEAT,
        }

    def has_labelled_contradiction(self) -> bool:
        """True if any LABELLED gold code contradicts the resolver output.

        Contradiction = scored cells where a human label disagrees with the
        prediction: ``wrong`` (mapped to the wrong concept), ``fn`` (gold
        RESOLVED but we said NO_MAP), ``fp_map`` (gold NO_MAP but we mapped).
        ``recall_miss`` (Zone-2 review-pending) is EXCLUDED by design. Fires at
        any ``labelled_count > 0``; full coverage is required only to GRANT the
        ACCEPTED verdict, never to start honoring labels. This — not the
        ACCEPTED verdict — is what gates ``sema fit --strict`` on gold.
        """
        m = self.score.distinct_code
        return (m.wrong + m.fn + m.fp_map) > 0

    def _ungraded_note(self) -> str:
        """Name the labelled-but-ungraded codes inline — a gap buried in JSON was
        indistinguishable from a complete run."""
        if not self.ungraded_codes:
            return ""
        listed = ", ".join(self.ungraded_codes[:10])
        more = "" if len(self.ungraded_codes) <= 10 else ", …"
        return f" — {len(self.ungraded_codes)} labelled but ungraded: {listed}{more}"

    def human_summary(self) -> str:
        m = self.score.distinct_code
        qualified = " ".join(f"[{q}]" for q in self.qualifiers)
        lines = [
            f"Mapping eval report — VERDICT: {self.verdict.value} {qualified}".rstrip(),
            f"  gold set: {self.snapshot_version or 'unversioned'}",
            f"  reason: {self.verdict_reason}",
            f"  coverage: labelled {self.labelled_count}/{self.total_codes} "
            f"({self.coverage_fraction:.1%})",
            f"  graded: {self.score.scored_codes}/{self.total_codes} of the frozen "
            f"tier scored{self._ungraded_note()}",
            "  distinct-code metrics:",
            f"    mapped_precision    = {_pct(m.mapped_precision)}",
            f"    mapped_recall       = {_pct(m.mapped_recall)}",
            f"    auto_resolution_rate= {_pct(m.auto_resolution_rate)}",
            f"    no_map_accuracy     = {_pct(m.no_map_accuracy)} (reported separately)",
            f"  strata: head {self.score.scored_codes} scored (gating) / "
            f"challenge {self.score.challenge_scored_codes} scored / "
            f"random tail {len(self.tail_sample_codes)} sampled, "
            f"{self.tail_sample_labelled} labelled, {self.score.tail_scored_codes} "
            f"scored, {self.tail_sample_unlabellable} carry no gold row yet "
            "(never gating)",
            f"  NOTE: {STRUCTURAL_PRECISION_CAVEAT}",
        ]
        return "\n".join(lines)
