"""§1.5(f) confusion-matrix math for the Slice-0 mapping gold set (US-002).

This module is the frozen home of the metric definitions referenced by US-012.
The unit of evaluation is the **distinct source code**; predicted decisions are
partitioned by the §1.5(c) Zone model and scored against an external gold label.

NOTE on the recall denominator: §1.5(f) writes
``mapped_recall = TP / (TP + WRONG + FN)`` and separately states that a Zone-2
prediction for a gold-``RESOLVED`` code "is a recall miss". We honour both by
keeping the denominator equal to *all* gold-``RESOLVED`` distinct codes
(``TP + WRONG + FN + recall_miss``); when no Zone-2 predictions exist (the
common Slice-0 case) this reduces exactly to the documented formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sema.models.planner.lifecycle import Status

# Frequency-bucket edges (row_count): codes are bucketed for per-bucket metrics.
_BUCKET_HIGH_MIN = 1000
_BUCKET_MEDIUM_MIN = 100


class GoldLabel(str, Enum):
    """External oracle label for a distinct source code.

    ``UNLABELLED`` marks a scaffolded code awaiting the human-label gate
    (US-002): it is never invented by Sema and is excluded from scoring.
    """

    RESOLVED = "RESOLVED"
    NO_MAP = "NO_MAP"
    UNLABELLED = "UNLABELLED"


class ResolutionStatus(str, Enum):
    """Per-decision resolution outcome (mirrors the value-mapping store)."""

    RESOLVED = "RESOLVED"
    NO_MAP = "NO_MAP"


class Zone(str, Enum):
    """§1.5(c) derived zone — computed, never stored."""

    ZONE_1 = "AUTO_ACCEPTED_RESOLVED"
    ZONE_2 = "REVIEW_OR_UNRESOLVED"
    ZONE_3 = "NO_MAP_ACCEPTED"


_ZONE1_STATUSES = frozenset({Status.auto_accepted, Status.human_pinned})
_ZONE2_STATUSES = frozenset(
    {Status.candidate, Status.review_pending, Status.rejected}
)


@dataclass(frozen=True)
class GoldRow:
    """One hand-labelled distinct source code (the gold-set artifact row)."""

    oncotree_code: str
    gold_concept_id: int | None
    gold_label: GoldLabel
    row_count: int
    notes: str = ""


@dataclass(frozen=True)
class Decision:
    """A predicted resolver decision for one distinct source code."""

    source_code: str
    concept_id: int | None
    status: Status
    resolution_status: ResolutionStatus
    no_map_reason: str | None = None


def derive_zone(
    status: Status,
    resolution_status: ResolutionStatus,
    concept_id: int | None,
) -> Zone:
    """Map (status, resolution_status, concept_id) to a §1.5(c) Zone."""
    if resolution_status is ResolutionStatus.NO_MAP and concept_id is None:
        return Zone.ZONE_3
    if (
        status in _ZONE1_STATUSES
        and resolution_status is ResolutionStatus.RESOLVED
        and concept_id is not None
    ):
        return Zone.ZONE_1
    return Zone.ZONE_2


def classify_cell(gold: GoldRow, zone: Zone, concept_id: int | None) -> str:
    """Return the §1.5(f) confusion-matrix cell name for one prediction.

    Cells: tp, wrong, fn, recall_miss (gold RESOLVED) and tn, fp_map, na
    (gold NO_MAP). ``na`` = a gold-NO_MAP code predicted Zone-2 (n/a).
    """
    if gold.gold_label is GoldLabel.RESOLVED:
        if zone is Zone.ZONE_1:
            return "tp" if concept_id == gold.gold_concept_id else "wrong"
        if zone is Zone.ZONE_3:
            return "fn"
        return "recall_miss"
    if zone is Zone.ZONE_1:
        return "fp_map"
    if zone is Zone.ZONE_3:
        return "tn"
    return "na"


_CELLS = ("tp", "wrong", "fn", "recall_miss", "tn", "fp_map", "na")


@dataclass
class ConfusionMatrix:
    """§1.5(f) confusion matrix with derived metrics (None on empty denom)."""

    tp: float = 0.0
    wrong: float = 0.0
    fn: float = 0.0
    recall_miss: float = 0.0
    tn: float = 0.0
    fp_map: float = 0.0
    na: float = 0.0

    def add(self, cell: str, weight: float = 1.0) -> None:
        if cell not in _CELLS:
            raise ValueError(f"unknown confusion cell: {cell!r}")
        setattr(self, cell, getattr(self, cell) + weight)

    @property
    def scored(self) -> float:
        """All labelled units scored (distinct count or row-weighted sum)."""
        return self.tp + self.wrong + self.fn + self.recall_miss + self.tn + self.fp_map + self.na

    @property
    def zone1(self) -> float:
        return self.tp + self.wrong + self.fp_map

    @property
    def mapped_precision(self) -> float | None:
        denom = self.tp + self.wrong + self.fp_map
        return None if denom == 0 else self.tp / denom

    @property
    def mapped_recall(self) -> float | None:
        denom = self.tp + self.wrong + self.fn + self.recall_miss
        return None if denom == 0 else self.tp / denom

    @property
    def auto_resolution_rate(self) -> float | None:
        return None if self.scored == 0 else self.zone1 / self.scored

    @property
    def no_map_accuracy(self) -> float | None:
        denom = self.tn + self.fp_map
        return None if denom == 0 else self.tn / denom

    def as_dict(self) -> dict[str, float | None]:
        return {
            "tp": self.tp,
            "wrong": self.wrong,
            "fn": self.fn,
            "recall_miss": self.recall_miss,
            "tn": self.tn,
            "fp_map": self.fp_map,
            "na": self.na,
            "scored": self.scored,
            "mapped_precision": self.mapped_precision,
            "mapped_recall": self.mapped_recall,
            "auto_resolution_rate": self.auto_resolution_rate,
            "no_map_accuracy": self.no_map_accuracy,
        }


def frequency_bucket(row_count: int) -> str:
    """Bucket a distinct code by its source row_count (high/medium/low)."""
    if row_count >= _BUCKET_HIGH_MIN:
        return "high"
    if row_count >= _BUCKET_MEDIUM_MIN:
        return "medium"
    return "low"
