"""Run metrics (design Decision 13), computed from the stage results and the registry row."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from sema.intake.eligibility_spec_utils import RolesResult
from sema.intake.landing import LandResult
from sema.intake.registry import RunRecord
from sema.intake.shapes import ignore
from sema.intake.structure import StructureResult
from sema.intake.verify.base import Ambiguity

__all__ = ["RunMetrics", "run_metrics"]


class RunMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relations: int
    ignored: dict[str, int]
    bound: int
    bound_fraction: float
    open_structure: int
    open_roles: int
    proposals_made: int
    proposals_bound: int
    model_calls: int
    tokens: int
    cost: float
    value_asks: int


def run_metrics(
    land: LandResult,
    structure: StructureResult,
    roles: RolesResult | None,
    record: RunRecord,
    *,
    proposals_made: int = 0,
    proposals_bound: int = 0,
    value_asks: int = 0,
) -> RunMetrics:
    relations = len(land.relations)
    bound = len({name for entry in structure.bound for name in entry.binding.relations})
    ignored = Counter(
        binding.slots[ignore.REASON] for binding in structure.ignored for _ in binding.relations
    )
    # A no_header ignore stays scored so ignoring a real table cannot raise the fraction.
    scored = relations - sum(n for reason, n in ignored.items() if reason != ignore.NO_HEADER)
    return RunMetrics(
        relations=relations,
        ignored=dict(ignored),
        bound=bound,
        bound_fraction=bound / scored if scored > 0 else 0.0,
        open_structure=_open(structure.ambiguities),
        open_roles=_open(roles.ambiguities) if roles is not None else 0,
        proposals_made=proposals_made,
        proposals_bound=proposals_bound,
        model_calls=record.model_calls,
        tokens=record.tokens,
        cost=record.cost,
        value_asks=value_asks,
    )


def _open(ambiguities: Iterable[Ambiguity]) -> int:
    """Open as the reader counts it: a human ``none`` has no choice but a source."""
    return sum(1 for entry in ambiguities if entry.choice is None and entry.source is None)
