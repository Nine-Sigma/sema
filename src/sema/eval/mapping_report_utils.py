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

from dataclasses import dataclass
from enum import Enum
from typing import Any

from sema.eval.mapping_goldset import GoldSetReport
from sema.eval.mapping_goldset_utils import (
    ConfusionMatrix,
    Decision,
    ResolutionStatus,
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
    """Whether the run may be called 'accepted' — never self-certified."""

    ACCEPTED = "accepted"
    PROVISIONAL_NOT_ACCEPTED = "provisional — not accepted"
    RUNNING_NOT_ACCEPTED = "running, not accepted"


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
) -> tuple[AcceptanceVerdict, str]:
    """Apply the §1.5(f) acceptance gate to one confusion matrix + coverage."""
    if coverage_fraction < 1.0:
        return (
            AcceptanceVerdict.PROVISIONAL_NOT_ACCEPTED,
            f"labelled gold coverage {coverage_fraction:.1%} < 100% of observed "
            "distinct codes; the US-002 human-label gate is incomplete",
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "verdict_reason": self.verdict_reason,
            "coverage": {
                "labelled": self.labelled_count,
                "total": self.total_codes,
                "fraction": self.coverage_fraction,
                "unlabelled_codes": list(self.unlabelled_codes),
            },
            "thresholds": {
                "mapped_precision": MIN_MAPPED_PRECISION,
                "auto_resolution_rate": MIN_AUTO_RESOLUTION_RATE,
            },
            "metrics": self.score.as_dict(),
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

    def human_summary(self) -> str:
        m = self.score.distinct_code
        lines = [
            f"Mapping eval report — VERDICT: {self.verdict.value}",
            f"  reason: {self.verdict_reason}",
            f"  coverage: labelled {self.labelled_count}/{self.total_codes} "
            f"({self.coverage_fraction:.1%})",
            "  distinct-code metrics:",
            f"    mapped_precision    = {_pct(m.mapped_precision)}",
            f"    mapped_recall       = {_pct(m.mapped_recall)}",
            f"    auto_resolution_rate= {_pct(m.auto_resolution_rate)}",
            f"    no_map_accuracy     = {_pct(m.no_map_accuracy)} (reported separately)",
            f"  NOTE: {STRUCTURAL_PRECISION_CAVEAT}",
        ]
        return "\n".join(lines)
