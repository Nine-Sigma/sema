"""Roles resolution state and answer readers: what a role map or reference answer states.

``RolesRun`` holds the choices accepted so far and why each other entry stays open. A role
map answer states a choice for each open role; a lexicon claim answered with another
column is contested and opens with both answers (S5 decision 18c); a claim code, memory,
or a decision settled is kept whatever the map says. Under agreement two answers merge
per subject first (``resolve_agreement_utils``).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field

from sema.binding.spec import ConceptSourceSpec
from sema.intake.roles.eligibility import run_roles
from sema.intake.roles.eligibility_spec_utils import RolesResult
from sema.intake.roles.eligibility_utils import (
    EligibilitySource,
    Embedder,
    RoleEligibility,
    StatedChoice,
)
from sema.intake.calls.model_calls import ModelCalls
from sema.intake.resolve.output_models_utils import ABSTAIN, NONE, Choice
from sema.intake.land.profile import ProfileTools
from sema.intake.resolve.resolve_agreement_utils import AGREED, Agreement, Reading
from sema.intake.resolve.resolve_roles_utils import RoleAsk, ranked_first, unasked_reason
from sema.intake.structure.shapes.base import RelationProfile
from sema.intake.structure.structure import StructureResult
from sema.intake.verify.base import Ambiguity, Failure

MODEL: EligibilitySource = "model"
LEXICON: EligibilitySource = "lexicon"
CONTESTED = "contested: the lexicon claims `{}`; the model names `{}` ({})"


@dataclass
class RolesRun:
    """Choices accepted so far by subject, with each one's call key, reason, and findings;
    the reason every entry the model left open stays open, the answers ranked first on
    each it failed or the two asks disagreed on, each contested lexicon claim, the subjects
    two asks agreed on, and each relation's disagreement count."""

    structure: StructureResult
    profiles: list[RelationProfile]
    tools: ProfileTools
    calls: ModelCalls
    hint: ConceptSourceSpec | None
    embedder: Embedder | None
    accepted: dict[str, StatedChoice] = field(default_factory=dict)
    recorded: dict[str, tuple[str, str]] = field(default_factory=dict)
    left: dict[str, str] = field(default_factory=dict)
    ranked: dict[str, tuple[str, ...]] = field(default_factory=dict)
    findings: dict[str, tuple[str, ...]] = field(default_factory=dict)
    contested: dict[str, str] = field(default_factory=dict)
    agreed: set[str] = field(default_factory=set)
    disagreements: dict[str, int] = field(default_factory=dict)

    def roles(
        self,
        extra: Mapping[str, StatedChoice] | None = None,
        contested: Collection[str] = (),
    ) -> RolesResult:
        choices = {**self.accepted, **(extra or {})}
        return run_roles(
            self.structure, self.profiles, self.tools, hint=self.hint,
            embedder=self.embedder, choices=choices, contested={*self.contested, *contested},
        )

    def accept(self, attempt: RoleAttempt, stated: Mapping[str, StatedChoice]) -> None:
        self.accepted.update(stated)
        self.contested.update(attempt.contested)
        self.agreed.update(attempt.agreed & stated.keys())
        self.ranked.update(attempt.disagreed)
        for subject, reason in attempt.reasons.items():
            if subject in stated:
                self.recorded[subject] = (attempt.key, reason)
                self.findings[subject] = attempt.findings.get(subject, ())
            elif subject not in attempt.stated:
                self.left[subject] = reason

    def disagree(self, relation: str, attempt: RoleAttempt) -> None:
        if self.calls.agreement:
            count = self.disagreements.get(relation, 0)
            self.disagreements[relation] = count + len(attempt.disagreed)


@dataclass(frozen=True)
class RoleAttempt:
    stated: dict[str, StatedChoice]
    reasons: dict[str, str]
    key: str
    failures: tuple[Failure, ...] = ()
    findings: dict[str, tuple[str, ...]] = field(default_factory=dict)
    contested: dict[str, str] = field(default_factory=dict)
    problem: str | None = None
    error: str | None = None
    agreed: frozenset[str] = frozenset()
    disagreed: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def merged(
        cls,
        agreement: Agreement,
        key: str,
        failures: tuple[Failure, ...],
        findings: dict[str, tuple[str, ...]],
    ) -> RoleAttempt:
        found = agreement.reading
        return cls(
            found.stated, found.reasons, key, failures, findings, found.contested,
            agreed=agreement.agreed, disagreed=agreement.disagreed,
        )


def read_answer(
    ask: RoleAsk, entries: Mapping[str, RoleEligibility], choices: Sequence[Choice]
) -> Reading:
    if ask.references:
        return Reading(*read_references(ask, choices))
    return Reading(*read_map(ask, entries, choices))


def read_references(
    ask: RoleAsk, choices: Sequence[Choice]
) -> tuple[dict[str, StatedChoice], dict[str, str], dict[str, str]]:
    """Each reference answer that is not ``abstain``, and every entry's reason."""
    paired = list(zip(ask.entries, choices, strict=True))
    stated = {
        entry.subject: StatedChoice(choice=choice, source=MODEL, confidence=confidence)
        for entry, (choice, confidence, _) in paired
        if choice != ABSTAIN
    }
    return stated, {entry.subject: reason for entry, (_, _, reason) in paired}, {}


def read_map(
    ask: RoleAsk, entries: Mapping[str, RoleEligibility], choices: Sequence[Choice]
) -> tuple[dict[str, StatedChoice], dict[str, str], dict[str, str]]:
    """The model's answer per open role; a claim the code or a decision settled is kept,
    and a lexicon claim answered with another column is contested."""
    stated: dict[str, StatedChoice] = {}
    reasons: dict[str, str] = {}
    contested: dict[str, str] = {}
    for found, (choice, confidence, reason) in zip(ask.fields(), choices, strict=True):
        entry = entries.get(found.subject)
        if entry is None:
            continue
        reasons[found.subject] = reason
        if entry.source == LEXICON:
            if choice not in (entry.choice, ABSTAIN):
                text = CONTESTED.format(entry.choice, choice, reason)
                contested.update(dict.fromkeys(_with_kin(ask, found.subject), text))
            continue
        offered = choice != NONE or NONE in entry.members
        if entry.source is None and choice != ABSTAIN and offered:
            stated[found.subject] = StatedChoice(
                choice=choice, source=MODEL, confidence=confidence
            )
    return stated, reasons, contested


def _with_kin(ask: RoleAsk, subject: str) -> tuple[str, ...]:
    """The subject and the same role on each kin relation, which inherits the map."""
    role = subject.removeprefix(ask.relation)
    return (subject, *(f"{kin}{role}" for kin in ask.kin))


def recorded_entry(run: RolesRun, ambiguity: Ambiguity, asks: Sequence[RoleAsk]) -> Ambiguity:
    subject = ambiguity.subject
    if ambiguity.choice is None and subject in run.contested:
        return ambiguity.model_copy(update={"reason": run.contested[subject]})
    if ambiguity.source == MODEL and subject in run.recorded:
        key, reason = run.recorded[subject]
        hints = (*ambiguity.hints, *run.findings.get(subject, ()))
        agreement = AGREED if subject in run.agreed else None
        return ambiguity.model_copy(
            update={"call_key": key, "reason": reason, "hints": hints, "agreement": agreement}
        )
    if ambiguity.choice is None and subject in run.left:
        update: dict[str, object] = {"reason": run.left[subject]}
        if subject in run.ranked:
            update["members"] = ranked_first(ambiguity.members, run.ranked[subject])
        return ambiguity.model_copy(update=update)
    if ambiguity.choice is not None or ambiguity.source is not None or ambiguity.reason:
        return ambiguity
    unasked = unasked_reason(ambiguity, asks, run.accepted)
    return ambiguity if unasked is None else ambiguity.model_copy(update={"reason": unasked})
