"""Shared setup for test_heldout_gate.py and the tests that reuse it."""

from __future__ import annotations

from dataclasses import dataclass

from sema.intake.run.metrics import RunMetrics


@dataclass(frozen=True)
class GateThreshold:
    min_bound_fraction: float
    max_open_choices: int


@dataclass(frozen=True)
class GateMiss:
    metric: str
    value: float
    threshold: float

    def __str__(self) -> str:
        return f"{self.metric} {self.value} misses threshold {self.threshold}"


# Decided by Dean 2026-10-21 and committed before any model-backed run over either source
# (task 14.1). Open choices are counted at Review, before any decisions file.
GATE_THRESHOLDS = {
    "heldout": GateThreshold(min_bound_fraction=0.75, max_open_choices=2),
    "heldout2": GateThreshold(min_bound_fraction=1.0, max_open_choices=3),
    # Decided by Dean 2026-10-22 (S4 task 3.1), committed before any model-backed run over
    # either source.
    "heldout3": GateThreshold(min_bound_fraction=1.0, max_open_choices=2),
    "heldout4": GateThreshold(min_bound_fraction=1.0, max_open_choices=3),
    # Set by the supervisor 2026-10-23 (S5 decision 24), committed before any model-backed run.
    "heldout5": GateThreshold(min_bound_fraction=1.0, max_open_choices=4),
}


def gate_misses(metrics: RunMetrics, threshold: GateThreshold) -> tuple[GateMiss, ...]:
    misses: list[GateMiss] = []
    if metrics.bound_fraction < threshold.min_bound_fraction:
        misses.append(
            GateMiss("bound_fraction", metrics.bound_fraction, threshold.min_bound_fraction)
        )
    open_choices = metrics.open_structure + metrics.open_roles
    if open_choices > threshold.max_open_choices:
        misses.append(GateMiss("open_choices", open_choices, threshold.max_open_choices))
    return tuple(misses)
