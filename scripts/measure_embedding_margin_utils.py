"""Cases, ground truth, and the floor rule for the embedding margin measurement (task 8.3).

The floor is ceil to two decimals of the widest margin over a wrong top choice, plus
0.01, so no wrong choice on the measurement set settles; with no wrong choice it is
ceil to two decimals of the narrowest correct margin.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import RelationRef, StatusRole
from sema.intake import lexicon
from sema.intake.eligibility_utils import Embedder, RoleEligibility, embedding_rank
from sema.intake.shapes.base import RelationProfile

HELD_OUT_DIRS = frozenset({"heldout", "heldout2"})
FLOOR_STEP = 0.01
FLOOR_DECIMALS = 2
# Absorbs float noise such as 0.07 * 100 == 7.000000000000001 before the ceiling.
CEIL_PRECISION = 9

RoleKey = tuple[str, str]


class HeldOutSourceError(ValueError):
    """A held-out source is never an input to the floor."""


@dataclass(frozen=True)
class MarginCase:
    fixture: str
    relation: str
    role: str
    members: tuple[str, ...]
    top: str
    expected: str | None
    margin: float

    @property
    def skipped(self) -> bool:
        return self.expected is None or self.expected not in self.members

    @property
    def correct(self) -> bool:
        return not self.skipped and self.top == self.expected

    @property
    def verdict(self) -> str:
        if self.skipped:
            return "skipped"
        return "correct" if self.correct else "wrong"


def refuse_heldout(path: Path | str) -> Path:
    resolved = Path(path).resolve()
    found = sorted(HELD_OUT_DIRS.intersection(resolved.parts))
    if found:
        raise HeldOutSourceError(f"{resolved} is under held-out {found[0]}; never measured")
    return resolved


def _ceil(value: float) -> float:
    scale: int = 10**FLOOR_DECIMALS
    return math.ceil(round(value * scale, CEIL_PRECISION)) / scale


def select_floor(cases: Iterable[MarginCase]) -> float:
    counted = [case for case in cases if not case.skipped]
    if not counted:
        raise ValueError("no case has an eligible expected column; the floor is unmeasured")
    wrong = [case.margin for case in counted if not case.correct]
    if wrong:
        return round(_ceil(max(wrong)) + FLOOR_STEP, FLOOR_DECIMALS)
    return round(_ceil(min(case.margin for case in counted)), FLOOR_DECIMALS)


def _role_sections(spec: ConceptSourceSpec) -> Iterator[tuple[str, BaseModel]]:
    for name in type(spec).model_fields:
        value = getattr(spec, name)
        for section in value if isinstance(value, list) else [value]:
            relation = getattr(section, "relation", None)
            roles = getattr(section, "roles", None)
            if isinstance(relation, RelationRef) and isinstance(roles, BaseModel):
                yield relation.name, roles
    model = spec.entity_model
    if model is None:
        return
    for entry in model.entities or ():
        yield entry.relation.name, model.roles


def _column(value: object) -> str | None:
    if isinstance(value, StatusRole):
        return value.column
    return value if isinstance(value, str) else None


def expected_columns(spec: ConceptSourceSpec) -> dict[RoleKey, str]:
    """The first section naming a role on a relation is its ground truth."""
    truth: dict[RoleKey, str] = {}
    for relation, roles in _role_sections(spec):
        for role in type(roles).model_fields:
            column = _column(getattr(roles, role))
            if column is not None:
                truth.setdefault((relation, role), column)
    return truth


def margin_cases(
    fixture: str,
    eligibilities: Iterable[RoleEligibility],
    profiles: Sequence[RelationProfile],
    expected: Mapping[RoleKey, str],
    embedder: Embedder,
) -> list[MarginCase]:
    """One case per relation-level role whose predicate set has two or more columns."""
    by_name = {profile.ref.name: profile for profile in profiles}
    entries = {entry.name: entry for entry in lexicon.ROLES}
    found = []
    for eligibility in eligibilities:
        entry = entries.get(eligibility.role)
        profile = by_name.get(eligibility.relation)
        if eligibility.subject is not None or entry is None or profile is None:
            continue
        columns = [c for c in profile.columns if c.name in eligibility.members]
        if len(columns) < 2:
            continue
        ranked = embedding_rank(embedder, entry, columns)
        found.append(MarginCase(
            fixture=fixture,
            relation=eligibility.relation,
            role=eligibility.role,
            members=eligibility.members,
            top=ranked.rank[0][0],
            expected=expected.get((eligibility.relation, eligibility.role)),
            margin=ranked.rank[0][1] - ranked.rank[1][1],
        ))
    return found
