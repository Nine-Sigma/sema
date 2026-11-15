"""The view contract: a recordable spec, the protocol, and the refusal (design Decision 4).

A view turns a landed relation a template cannot read into a derived staging relation
it can. ``verify`` checks the landed relation the spec names; ``apply`` materializes the
view and returns its profile, ready for a template's ``verify``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sema.intake.land.profile import ProfileTools
from sema.intake.structure.shapes.base import RelationProfile
from sema.intake.verify.base import Failure, Verdict


class ViewSpec(BaseModel):
    """``relation`` is the landed relation's ref name; ``columns`` keep their stated order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    view: str
    relation: str
    columns: tuple[str, ...] = Field(min_length=1)

    @field_validator("columns")
    @classmethod
    def _named_once(cls, columns: tuple[str, ...]) -> tuple[str, ...]:
        if repeated := sorted({c for c in columns if columns.count(c) > 1}):
            raise ValueError(f"columns {repeated} are named more than once")
        return columns


class ViewRefusedError(ValueError):
    """A spec its view's ``verify`` fails; the failures say why."""

    def __init__(self, spec: ViewSpec, failures: Sequence[Failure]) -> None:
        self.spec = spec
        self.failures = tuple(failures)
        checks = ", ".join(f"{f.check} ({f.column or f.relation})" for f in self.failures)
        super().__init__(f"view {spec.view} over {spec.relation} is refused: {checks}")


class View(Protocol):
    ID: str

    def verify(
        self, spec: ViewSpec, group: Sequence[RelationProfile], tools: ProfileTools
    ) -> Verdict: ...

    def apply(
        self, spec: ViewSpec, group: Sequence[RelationProfile], tools: ProfileTools
    ) -> RelationProfile: ...
