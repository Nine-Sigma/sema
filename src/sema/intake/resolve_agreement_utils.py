"""Agreement between two asks: the Roles confidence signal (S5 decisions 4 and 16).

With agreement on, a role map or reference ask is made twice, the second listing the
columns in the relation's seeded order. Per subject, two answers naming the same column
settle with the lower confidence and ``agreement: 2/2``. Answers that differ, including
one answer against an abstention, leave the subject open with both answers ranked by
confidence and both reasons, and count once toward the relation's disagreements.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from sema.intake.eligibility_utils import StatedChoice

AGREED = "2/2"
DISAGREED = "the two asks disagree: {}"
NO_CHOICE = "no column"


@dataclass(frozen=True)
class Reading:
    """What one answer states per subject, every subject's reason, and the lexicon claims
    it contests."""

    stated: dict[str, StatedChoice]
    reasons: dict[str, str]
    contested: dict[str, str]


@dataclass(frozen=True)
class Agreement:
    """The merged reading; ``disagreed`` holds each open subject's answers, ranked."""

    reading: Reading
    agreed: frozenset[str] = frozenset()
    disagreed: dict[str, tuple[str, ...]] = field(default_factory=dict)


def agree(readings: Sequence[Reading], members: Mapping[str, Sequence[str]]) -> Agreement:
    """One reading passes through; two merge per subject. ``members`` bounds the answers
    an open entry ranks, so a name outside the entry is stated in the reason only."""
    if len(readings) == 1:
        return Agreement(readings[0])
    first, second = readings
    stated: dict[str, StatedChoice] = {}
    reasons = dict(first.reasons)
    disagreed: dict[str, tuple[str, ...]] = {}
    for subject in dict.fromkeys([*first.reasons, *second.reasons]):
        one, two = first.stated.get(subject), second.stated.get(subject)
        if one is None and two is None:
            continue
        if one is not None and two is not None and one.choice == two.choice:
            lower = min(one.confidence, two.confidence)
            stated[subject] = one.model_copy(update={"confidence": lower})
            continue
        answers = _ranked([(one, first.reasons.get(subject)), (two, second.reasons.get(subject))])
        reasons[subject] = DISAGREED.format("; ".join(_said(*answer) for answer in answers))
        offered = members.get(subject, ())
        named = (choice.choice for choice, _ in answers if choice is not None)
        disagreed[subject] = tuple(name for name in named if name in offered)
    contested = {**second.contested, **first.contested}
    return Agreement(Reading(stated, reasons, contested), frozenset(stated), disagreed)


Answer = tuple[StatedChoice | None, str | None]


def _ranked(answers: Sequence[Answer]) -> list[Answer]:
    """Highest confidence first; an abstention ranks last; a tie keeps the ask order."""
    return sorted(answers, key=lambda answer: -answer[0].confidence if answer[0] else 1.0)


def _said(choice: StatedChoice | None, reason: str | None) -> str:
    if choice is None:
        return f"{NO_CHOICE} ({reason or 'no reason given'})"
    return f"`{choice.choice}` at {choice.confidence:.2f} ({reason or 'no reason given'})"
