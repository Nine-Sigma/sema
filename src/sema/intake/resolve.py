"""Resolve structure and roles: the model settles open entries among the members code proved.

Entries go out in batches of at most ``MAX_STRUCTURE_BATCH``, one call each through the
replayed, capped model-call path. A choice is written onto its entry with ``source: model``,
the confidence, and the call key, and its binding is bound with ``source: model``. The
Structure verifier re-runs on every answer; a batch whose choices fail is re-asked with the
failed checks and the bindings in force, at most ``max_retries`` times, and then only the
choices spanning a failing relation are left open, each with the failure as its reason.

Column roles go out as one role map per relation signature (S5 decision 1), then
reference entries one call per relation. The map names any column per role; code verifies
it after (decision 2): a name outside the relation, two roles on one column, or a failed
data fact is re-asked once with every failure listed, and then the failing roles stay
open with the model's choice ranked first. A finding never fails a choice; it is recorded
on the entry. A lexicon claim the model contests opens with both answers (decision 18c).
Choices re-enter Roles as stated choices, so a relation sharing a signature inherits them
with ``source: memory``. One retry serves every stage. With agreement on, each role ask
is made twice, the second over the columns in the relation's seeded order; the two
answers merge per role before the verifier runs (decisions 4 and 16).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace

from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import Provider
from sema.intake.eligibility_spec_utils import RolesResult
from sema.intake.eligibility_text_utils import STAGE as ROLES_STAGE
from sema.intake.eligibility_utils import Embedder, RoleEligibility, StatedChoice
from sema.intake.landing import LandResult
from sema.intake.model_calls import ModelCalls
from sema.intake.output_models import (
    MAX_STRUCTURE_BATCH,
    entry_choices,
    structure_batch_model,
)
from sema.intake.profile import ProfileTools
from sema.intake.resolve_budget_utils import fitted_structure_prompt
from sema.intake.resolve_agreement_utils import Reading, agree
from sema.intake.resolve_answer_utils import (
    RoleAttempt,
    RolesRun,
    read_answer,
    recorded_entry,
)
from sema.intake.resolve_prompt_utils import RoleEvidence
from sema.intake.resolve_roles_utils import (
    RoleAsk,
    chosen_notes,
    column_asks,
    fitted_roles_prompt,
    map_entries,
    map_roles,
    reference_asks,
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
    reopened,
    retried,
    settle_choices,
)
from sema.intake.shapes.base import RelationProfile, profile_relations
from sema.intake.shapes.precedence import STAGE
from sema.intake.structure import BoundEntry, StructureResult
from sema.intake.structure_utils import describe_failure
from sema.intake.verify.base import Ambiguity, Failure
from sema.intake.verify.roles import COLUMN_EXISTS, ROLE_CHECKS, verify_roles
from sema.intake.verify.structure import verify_structure
from sema.llm_client import LLMStageError
from sema.log import logger

__all__ = ["DEFAULT_MAX_RETRIES", "resolve_roles", "resolve_structure"]

DEFAULT_MAX_RETRIES = 1
MISALIGNED = "the answer's confidences or reasons do not list one item per entry"
CALL_FAILED = "the model call failed"


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
    output_model = structure_batch_model([entry.ambiguity.members for entry in batch])
    prompt = fitted_structure_prompt(
        run.tools, run.profiles, run.bound, batch, failed, output_model
    )
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
) -> RolesResult:
    """``hint`` and ``embedder`` are the ones Roles ran with."""
    profiles = profile_relations(land, tools, provider)
    run = RolesRun(structure, profiles, tools, calls, hint, embedder)
    by_name = {profile.ref.name: profile for profile in profiles}
    asks = column_asks(roles, by_name)
    for ask in asks:
        _settle_roles(run, ask, max_retries)
    references = reference_asks(run.roles())
    for ask in references:
        _settle_roles(run, ask, max_retries)
    result = run.roles()
    logger.info(
        "resolve roles: {} role-map calls and {} reference calls, {} choices settled, {} left"
        " open, {} lexicon claims contested",
        len(asks), len(references), len(run.recorded), len(run.left), len(run.contested),
    )
    asked = (*asks, *references)
    entries = tuple(recorded_entry(run, a, asked) for a in result.ambiguities)
    return replace(result, ambiguities=entries, disagreements=run.disagreements)


def _settle_roles(run: RolesRun, ask: RoleAsk, max_retries: int) -> None:
    current = run.roles()
    names = spanned(current, ask.relation)
    relations = [p for p in run.profiles if p.ref.name in names]
    entries = map_entries(current, ask.relation)
    evidence = RoleEvidence(run.tools, relations, ask.relation, list(entries.values()))

    def _ask(failed: Sequence[Sequence[str]]) -> RoleAttempt:
        orders = (False, True) if run.calls.agreement else (False,)
        prompts = [
            fitted_roles_prompt(ask, current, evidence, failed, shuffled=shuffled)
            for shuffled in orders
        ]
        return _ask_roles(run, ask, entries, prompts)

    attempt, passed = retried(
        _ask,
        lambda found: _role_notes(ask, found),
        max_retries,
        ended=lambda found: found.error is not None,
    )
    run.disagree(ask.relation, attempt)
    if passed:
        run.accept(attempt, attempt.stated)
        return
    _leave_roles_open(run, ask, attempt)


def _ask_roles(
    run: RolesRun, ask: RoleAsk, entries: Mapping[str, RoleEligibility], prompts: Sequence[str]
) -> RoleAttempt:
    """One answer per prompt, merged; the entry records the first ask's call key."""
    output_model = ask.output_model()
    key = run.calls.call_key(prompts[0], output_model)
    readings: list[Reading] = []
    for prompt in prompts:
        try:
            answer = run.calls.call(ROLES_STAGE, prompt, output_model)
        except LLMStageError as error:
            logger.warning(f"Roles call {key} for {ask.relation} failed: {error}")
            return RoleAttempt({}, {}, key, problem=CALL_FAILED, error=str(error))
        choices = ask.choices(answer)
        if choices is None:
            return RoleAttempt({}, {}, key, problem=MISALIGNED)
        readings.append(read_answer(ask, entries, choices))
    members = {entry.subject: entry.ambiguity.members for entry in ask.entries}
    agreement = agree(readings, members)
    found = agreement.reading
    failures, findings = _verified(run, ask, found.stated, found.contested)
    return RoleAttempt.merged(agreement, key, failures, findings)


def _verified(
    run: RolesRun,
    ask: RoleAsk,
    stated: Mapping[str, StatedChoice],
    contested: Collection[str] = (),
) -> tuple[tuple[Failure, ...], dict[str, tuple[str, ...]]]:
    """The verifier's role-check failures on a chosen column of the asked relation or kin,
    and the findings on each stated choice's column."""
    columns = ask.columns(stated)
    if not columns:
        return (), {}
    relations = (ask.relation, *ask.kin)
    roles = run.roles(stated, contested)
    verdict = verify_roles(roles, run.tools, spec=None, relations=relations)
    failures = tuple(
        found for found in verdict.failures
        if found.check in ROLE_CHECKS and found.relation in relations and found.column in columns
    )
    findings = {
        subject: tuple(
            f"finding: {found.name} on {found.subject} ({found.value})"
            for found in verdict.findings
            if found.subject == f"{ask.relation}.{choice.choice}"
        )
        for subject, choice in stated.items()
    }
    return failures, {subject: named for subject, named in findings.items() if named}


def _role_notes(ask: RoleAsk, attempt: RoleAttempt) -> list[str]:
    """Nothing for a passing attempt; else each failed check with the choices it names."""
    if attempt.problem is not None:
        return [attempt.problem]
    chosen = {subject: stated.choice for subject, stated in attempt.stated.items()}
    roles = map_roles(ask)
    notes = []
    for found in attempt.failures:
        named = {s: c for s, c in chosen.items() if c == found.column}
        notes.append(f"{describe_failure(found)}; {chosen_notes(roles, named)}")
    return list(dict.fromkeys(notes))


def _leave_roles_open(run: RolesRun, ask: RoleAsk, attempt: RoleAttempt) -> None:
    """Drop every choice a failure names until the rest pass, then accept the rest; a
    dropped choice ranks first on its entry unless it named no column of the relation."""
    if attempt.problem is not None:
        reason = entry_reason(attempt.problem, attempt.error)
        run.left.update({entry.subject: reason for entry in ask.entries})
        return
    stated, failures = dict(attempt.stated), attempt.failures
    while failures:
        for subject, choice in list(stated.items()):
            failing = [f for f in failures if f.column == choice.choice]
            if not failing:
                continue
            run.left[subject] = "; ".join(dict.fromkeys(map(describe_failure, failing)))
            if any(f.check != COLUMN_EXISTS for f in failing):
                run.ranked[subject] = (choice.choice,)
            del stated[subject]
        failures, _ = _verified(run, ask, stated, attempt.contested)
    logger.info(
        "roles call for {} left {} choices open after retries; {} kept",
        ask.relation, len(attempt.stated) - len(stated), len(stated),
    )
    run.accept(attempt, stated)
