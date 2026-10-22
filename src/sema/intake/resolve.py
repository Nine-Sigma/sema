"""Resolve structure and roles: the model settles open entries among the members code proved.

Entries go out in batches of at most ``MAX_STRUCTURE_BATCH``, one call each through the
replayed, capped model-call path. A choice is written onto its entry with ``source: model``,
the confidence, and the call key, and its binding is bound with ``source: model``. The
Structure verifier re-runs on every answer; a batch whose choices fail is re-asked with the
failed checks and the bindings in force, at most ``max_retries`` times, and then only the
choices spanning a failing relation are left open, each with the failure as its reason.

Role entries go out one call per relation signature (column roles), then one call per
relation (reference entries). Choices re-enter Roles as stated choices, so a relation
sharing a signature inherits them with ``source: memory``; the Roles verifier re-proves
each answer over the asked relation and its kin, with the same retry bound.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import Provider
from sema.intake.eligibility import run_roles
from sema.intake.eligibility_spec_utils import RolesResult
from sema.intake.eligibility_text_utils import STAGE as ROLES_STAGE
from sema.intake.eligibility_utils import (
    EMBEDDING_MARGIN_FLOOR,
    EligibilitySource,
    Embedder,
    StatedChoice,
)
from sema.intake.landing import LandResult
from sema.intake.model_calls import ModelCalls
from sema.intake.output_models import (
    MAX_STRUCTURE_BATCH,
    entry_choices,
    structure_batch_model,
)
from sema.intake.output_models_utils import ABSTAIN
from sema.intake.profile import ProfileTools
from sema.intake.resolve_roles_utils import (
    RoleAsk,
    chosen_notes,
    column_asks,
    in_force,
    reference_asks,
    render_roles_prompt,
    spanned,
)
from sema.intake.resolve_utils import (
    OpenEntry,
    Settlement,
    entry_reason,
    failure_notes,
    member_bindings,
    open_entries,
    open_with,
    render_prompt,
    reopened,
    retried,
    schema_text,
    settle_choices,
)
from sema.intake.shapes.base import RelationProfile, profile_relations
from sema.intake.shapes.precedence import STAGE
from sema.intake.structure import BoundEntry, StructureResult
from sema.intake.structure_utils import describe_failure
from sema.intake.verify.base import Ambiguity, Failure
from sema.intake.verify.roles import ROLE_CHECKS, verify_roles
from sema.intake.verify.structure import verify_structure
from sema.llm_client import LLMStageError
from sema.log import logger

__all__ = ["DEFAULT_MAX_RETRIES", "resolve_roles", "resolve_structure"]

DEFAULT_MAX_RETRIES = 2
MISALIGNED = "the answer's confidences or reasons do not list one item per entry"
CALL_FAILED = "the model call failed"
MODEL: EligibilitySource = "model"


@dataclass
class _Run:
    land: LandResult
    tools: ProfileTools
    provider: Provider
    calls: ModelCalls
    profiles: dict[str, RelationProfile]
    structure: StructureResult
    ambiguities: list[Ambiguity]
    bound: list[BoundEntry]

    def proposal(self, settlement: Settlement | None = None) -> StructureResult:
        ambiguities = list(self.ambiguities)
        bound = list(self.bound)
        if settlement is not None:
            for position, ambiguity in settlement.updates.items():
                ambiguities[position] = ambiguity
            bound += settlement.chosen.values()
        update = {"ambiguities": tuple(ambiguities), "bound": tuple(bound)}
        return self.structure.model_copy(update=update)

    def apply(self, settlement: Settlement) -> None:
        for position, ambiguity in settlement.updates.items():
            self.ambiguities[position] = ambiguity
        self.bound += settlement.chosen.values()


@dataclass(frozen=True)
class _Attempt:
    settlement: Settlement | None
    failures: tuple[Failure, ...]
    problem: str | None = None
    error: str | None = None


def resolve_structure(
    structure: StructureResult,
    land: LandResult,
    tools: ProfileTools,
    calls: ModelCalls,
    *,
    provider: Provider = "staging",
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> StructureResult:
    profiles = profile_relations(land, tools, provider)
    pending = open_entries(structure, member_bindings(structure, profiles, tools))
    run = _Run(
        land=land,
        tools=tools,
        provider=provider,
        calls=calls,
        profiles={profile.ref.name: profile for profile in profiles},
        structure=structure,
        ambiguities=list(structure.ambiguities),
        bound=list(structure.bound),
    )
    for start in range(0, len(pending), MAX_STRUCTURE_BATCH):
        _settle(run, pending[start : start + MAX_STRUCTURE_BATCH], max_retries)
    result = run.proposal().with_settled()
    settled = sum(1 for entry in pending if run.ambiguities[entry.position].choice is not None)
    logger.info(
        "resolve structure: {} of {} open entries settled by the model, {} bound",
        settled, len(pending), len(result.bound) - len(structure.bound),
    )
    return result


def _settle(run: _Run, batch: Sequence[OpenEntry], max_retries: int) -> None:
    attempt, passed = retried(
        lambda failed: _ask(run, batch, failed),
        lambda found: _notes(batch, found),
        max_retries,
        ended=lambda found: found.error is not None,
    )
    if passed and attempt.settlement is not None:
        run.apply(attempt.settlement)
        return
    _leave_open(run, batch, attempt)


def _ask(
    run: _Run, batch: Sequence[OpenEntry], failed: Sequence[Sequence[str]]
) -> _Attempt:
    prompt = _prompt(run, batch, failed)
    output_model = structure_batch_model([entry.ambiguity.members for entry in batch])
    key = run.calls.call_key(prompt, output_model)
    try:
        answer = run.calls.call(STAGE, prompt, output_model)
    except LLMStageError as error:
        logger.warning(f"Structure batch call {key} failed: {error}")
        return _Attempt(None, (), CALL_FAILED, str(error))
    choices = entry_choices(answer, len(batch))
    if choices is None:
        return _Attempt(None, (), MISALIGNED)
    settlement = settle_choices(batch, choices, key)
    return _Attempt(settlement, _failures(run, settlement))


def _failures(run: _Run, settlement: Settlement) -> tuple[Failure, ...]:
    """The verifier's failures over relations this answer's choices span."""
    spanned = settlement.spanned()
    if not spanned:
        return ()
    proposal = run.proposal(settlement)
    verdict = verify_structure(proposal, run.land, run.tools, provider=run.provider)
    return tuple(found for found in verdict.failures if found.relation in spanned)


def _leave_open(run: _Run, batch: Sequence[OpenEntry], attempt: _Attempt) -> None:
    settlement = attempt.settlement
    if settlement is None:
        problem = entry_reason(attempt.problem or MISALIGNED, attempt.error)
        run.apply(
            Settlement("", {e.position: open_with(e.ambiguity, problem) for e in batch}, {})
        )
        return
    failures = attempt.failures
    while failures:
        settlement = reopened(settlement, failures)
        failures = _failures(run, settlement)
    logger.info(
        "structure batch left {} choices open after retries; {} bound",
        sum(1 for a in settlement.updates.values() if a.choice is None),
        len(settlement.chosen),
    )
    run.apply(settlement)


def _notes(batch: Sequence[OpenEntry], attempt: _Attempt) -> list[str]:
    """Nothing for a passing attempt; else what the next ask must be told."""
    if attempt.settlement is None:
        return [attempt.problem or MISALIGNED]
    return failure_notes(batch, attempt.settlement, attempt.failures)


def _prompt(
    run: _Run, batch: Sequence[OpenEntry], failed: Sequence[Sequence[str]]
) -> str:
    names = list(dict.fromkeys(name for entry in batch for name in entry.spanned()))
    spanned = set(names)
    in_force = [e for e in run.bound if spanned & set(e.binding.relations)]
    relations = [run.profiles[name] for name in names if name in run.profiles]
    return render_prompt(batch, in_force, relations, schema_text(run.tools, relations), failed)


@dataclass
class _RolesRun:
    """Choices accepted so far by subject, with each one's call key and reason, and the
    reason every entry the model left open stays open."""

    structure: StructureResult
    profiles: list[RelationProfile]
    tools: ProfileTools
    calls: ModelCalls
    hint: ConceptSourceSpec | None
    embedder: Embedder | None
    margin_floor: float | None
    accepted: dict[str, StatedChoice] = field(default_factory=dict)
    recorded: dict[str, tuple[str, str]] = field(default_factory=dict)
    left: dict[str, str] = field(default_factory=dict)

    def roles(self, extra: Mapping[str, StatedChoice] | None = None) -> RolesResult:
        choices = {**self.accepted, **(extra or {})}
        return run_roles(
            self.structure, self.profiles, self.tools, hint=self.hint,
            embedder=self.embedder, margin_floor=self.margin_floor, choices=choices,
        )

    def accept(self, attempt: _RoleAttempt, stated: Mapping[str, StatedChoice]) -> None:
        self.accepted.update(stated)
        for subject, reason in attempt.reasons.items():
            if subject in stated:
                self.recorded[subject] = (attempt.key, reason)
            elif subject not in attempt.stated:
                self.left[subject] = reason


@dataclass(frozen=True)
class _RoleAttempt:
    stated: dict[str, StatedChoice]
    reasons: dict[str, str]
    key: str
    failures: tuple[Failure, ...] = ()
    problem: str | None = None
    error: str | None = None


def resolve_roles(
    roles: RolesResult,
    structure: StructureResult,
    land: LandResult,
    tools: ProfileTools,
    calls: ModelCalls,
    *,
    provider: Provider = "staging",
    max_retries: int = DEFAULT_MAX_RETRIES,
    hint: ConceptSourceSpec | None = None,
    embedder: Embedder | None = None,
    margin_floor: float | None = EMBEDDING_MARGIN_FLOOR,
) -> RolesResult:
    """``hint``, ``embedder``, and ``margin_floor`` are the ones Roles ran with."""
    profiles = profile_relations(land, tools, provider)
    run = _RolesRun(structure, profiles, tools, calls, hint, embedder, margin_floor)
    by_name = {profile.ref.name: profile for profile in profiles}
    asks = column_asks(roles, by_name)
    for ask in asks:
        _settle_roles(run, ask, max_retries)
    references = reference_asks(run.roles())
    for ask in references:
        _settle_roles(run, ask, max_retries)
    result = run.roles()
    logger.info(
        "resolve roles: {} column calls and {} reference calls, {} choices settled, {} left open",
        len(asks), len(references), len(run.recorded), len(run.left),
    )
    return replace(result, ambiguities=tuple(_recorded(run, a) for a in result.ambiguities))


def _settle_roles(run: _RolesRun, ask: RoleAsk, max_retries: int) -> None:
    current = run.roles()
    names = spanned(current, ask.relation)
    relations = [p for p in run.profiles if p.ref.name in names]
    texts = schema_text(run.tools, relations)
    claims = in_force(current, ask.relation)

    def _ask(failed: Sequence[Sequence[str]]) -> _RoleAttempt:
        return _ask_roles(run, ask, render_roles_prompt(ask, claims, relations, texts, failed))

    attempt, passed = retried(
        _ask,
        lambda found: _role_notes(ask, found),
        max_retries,
        ended=lambda found: found.error is not None,
    )
    if passed:
        run.accept(attempt, attempt.stated)
        return
    _leave_roles_open(run, ask, attempt)


def _ask_roles(run: _RolesRun, ask: RoleAsk, prompt: str) -> _RoleAttempt:
    output_model = ask.output_model()
    key = run.calls.call_key(prompt, output_model)
    try:
        answer = run.calls.call(ROLES_STAGE, prompt, output_model)
    except LLMStageError as error:
        logger.warning(f"Roles call {key} for {ask.relation} failed: {error}")
        return _RoleAttempt({}, {}, key, problem=CALL_FAILED, error=str(error))
    choices = ask.choices(answer)
    if choices is None:
        return _RoleAttempt({}, {}, key, problem=MISALIGNED)
    paired = list(zip(ask.entries, choices, strict=True))
    stated = {
        entry.subject: StatedChoice(choice=choice, source=MODEL, confidence=confidence)
        for entry, (choice, confidence, _) in paired
        if choice != ABSTAIN
    }
    reasons = {entry.subject: reason for entry, (_, _, reason) in paired}
    return _RoleAttempt(stated, reasons, key, _role_failures(run, ask, stated))


def _role_failures(
    run: _RolesRun, ask: RoleAsk, stated: Mapping[str, StatedChoice]
) -> tuple[Failure, ...]:
    """The verifier's role-check failures on a chosen column of the asked relation or kin."""
    columns = {choice.choice for choice in stated.values()}
    if ask.references or not columns:
        return ()
    relations = (ask.relation, *ask.kin)
    verdict = verify_roles(run.roles(stated), run.tools, spec=None, relations=relations)
    return tuple(
        found for found in verdict.failures
        if found.check in ROLE_CHECKS and found.relation in relations and found.column in columns
    )


def _role_notes(ask: RoleAsk, attempt: _RoleAttempt) -> list[str]:
    """Nothing for a passing attempt; else each failed check with the choices it names."""
    if attempt.problem is not None:
        return [attempt.problem]
    chosen = {subject: stated.choice for subject, stated in attempt.stated.items()}
    notes = []
    for found in attempt.failures:
        named = {s: c for s, c in chosen.items() if c == found.column}
        notes.append(f"{describe_failure(found)}; {chosen_notes(ask.entries, named)}")
    return list(dict.fromkeys(notes))


def _leave_roles_open(run: _RolesRun, ask: RoleAsk, attempt: _RoleAttempt) -> None:
    """Drop every choice a failure names until the rest pass, then accept the rest."""
    if attempt.problem is not None:
        reason = entry_reason(attempt.problem, attempt.error)
        run.left.update({entry.subject: reason for entry in ask.entries})
        return
    stated, failures = dict(attempt.stated), attempt.failures
    while failures:
        for subject, choice in list(stated.items()):
            failing = [describe_failure(f) for f in failures if f.column == choice.choice]
            if failing:
                run.left[subject] = "; ".join(dict.fromkeys(failing))
                del stated[subject]
        failures = _role_failures(run, ask, stated)
    logger.info(
        "roles call for {} left {} choices open after retries; {} kept",
        ask.relation, len(attempt.stated) - len(stated), len(stated),
    )
    run.accept(attempt, stated)


def _recorded(run: _RolesRun, ambiguity: Ambiguity) -> Ambiguity:
    if ambiguity.source == MODEL and ambiguity.subject in run.recorded:
        key, reason = run.recorded[ambiguity.subject]
        return ambiguity.model_copy(update={"call_key": key, "reason": reason})
    if ambiguity.choice is None and ambiguity.subject in run.left:
        return ambiguity.model_copy(update={"reason": run.left[ambiguity.subject]})
    return ambiguity
