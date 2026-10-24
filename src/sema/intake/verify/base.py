"""The one verdict shape every stage verifier returns (design Decision 16).

A verifier returns ``pass`` or ``fail`` through ``Verdict.of``; ``escalate`` is set
only by the orchestrator once its retry budget is spent. A finding is a measured reading
reported beside the verdict; it never fails it (S5 decision 3). ``floor_classes`` names each
listed constant a ``gate`` (a data fact that removes a candidate or fails the verdict) or a
``finding`` (a measured reading that only annotates).
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

VerdictStatus = Literal["pass", "fail", "escalate"]
FloorValue = float | tuple[float, float] | None
FloorClass = Literal["gate", "finding"]
GATE: FloorClass = "gate"
FINDING: FloorClass = "finding"


def classed(gates: Iterable[str], findings: Iterable[str]) -> dict[str, FloorClass]:
    """The class of each constant a module applies; a name in both sets is a programming error."""
    both = sorted(set(gates) & set(findings))
    if both:
        raise ValueError(f"constants classed both gate and finding: {both}")
    return {**dict.fromkeys(gates, GATE), **dict.fromkeys(findings, FINDING)}


class Failure(BaseModel):
    model_config = ConfigDict(frozen=True)

    check: str
    relation: str
    column: str | None = None
    evidence: dict[str, Any]


class Finding(BaseModel):
    """``subject`` is the ``relation.column`` the reading is over."""

    model_config = ConfigDict(frozen=True)

    name: str
    subject: str
    value: float | str | None = None

    @field_validator("value")
    @classmethod
    def _decimal_digits(cls, value: float | str | None) -> float | str | None:
        """A ratio computed by subtraction (``1 - 16/20``) reads as the decimal it is."""
        if isinstance(value, float):
            return float(f"{value:.{sys.float_info.dig}g}")
        return value


class Ambiguity(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: str
    subject: str
    members: tuple[str, ...]
    hints: tuple[str, ...] = ()
    exclusions: dict[str, str] = {}
    choice: str | None = None
    confidence: float | None = None
    source: str | None = None
    call_key: str | None = None
    reason: str | None = None
    agreement: str | None = None

    @model_validator(mode="after")
    def _choice_is_a_sourced_member(self) -> Self:
        if self.choice is None:
            return self
        if self.choice not in self.members:
            raise ValueError(f"choice {self.choice!r} is not one of {list(self.members)}")
        if self.source is None:
            raise ValueError(f"choice {self.choice!r} for {self.subject} names no source")
        return self


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: VerdictStatus
    failures: tuple[Failure, ...] = ()
    ambiguities: tuple[Ambiguity, ...] = ()
    floors: dict[str, FloorValue] = {}
    floor_classes: dict[str, FloorClass] = {}
    findings: tuple[Finding, ...] = ()

    @model_validator(mode="after")
    def _status_matches_failures(self) -> Self:
        if self.status == "pass" and self.failures:
            raise ValueError("a passing verdict carries no failure")
        if self.status == "fail" and not self.failures:
            raise ValueError("a failing verdict names at least one failure")
        return self

    @model_validator(mode="after")
    def _classes_name_listed_floors(self) -> Self:
        unlisted = sorted(set(self.floor_classes) - set(self.floors))
        if unlisted:
            raise ValueError(f"floor classes name unlisted constants: {unlisted}")
        return self

    @classmethod
    def of(
        cls,
        *,
        failures: list[Failure] | tuple[Failure, ...],
        ambiguities: list[Ambiguity] | tuple[Ambiguity, ...],
        floors: dict[str, FloorValue],
        findings: list[Finding] | tuple[Finding, ...] = (),
        floor_classes: dict[str, FloorClass] | None = None,
    ) -> Verdict:
        status: VerdictStatus = "fail" if failures else "pass"
        return cls(
            status=status,
            failures=tuple(failures),
            ambiguities=tuple(ambiguities),
            floors=dict(floors),
            floor_classes=dict(floor_classes or {}),
            findings=tuple(findings),
        )
