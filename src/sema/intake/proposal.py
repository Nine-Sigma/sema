"""Propose layouts: the model binds a relation no template reads, and code verifies it.

Each proposal target (a ``no_template`` relation or one ignored as ``no_header``) gets one
call through the replayed, capped model-call path. A proposal names one template, its
relations and slots, and optionally one view. Code applies the view, runs the template's
``verify``, then the Structure verifier; a passing proposal is bound with ``source: model``.
Failures are re-asked with every earlier failure, at most ``max_retries`` times, and then
the relation is escalated for Review. An abstention is a layout finding; a request for
values is returned for the value ask and replays until values are approved, and a request
for columns already approved is an abstention (S5 decision 5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from sema.binding.spec_sections import Provider
from sema.intake.landing import LandResult
from sema.intake.model_calls import ModelCalls
from sema.intake.output_models_utils import ABSTAIN, NONE
from sema.intake.profile import ProfileTools
from sema.intake.profile_utils import NeedsValues
from sema.intake.resolve import DEFAULT_MAX_RETRIES
from sema.intake.resolve_utils import ShownValues, retried
from sema.intake.proposal_utils import (
    binding_of,
    layout_values,
    proposable,
    proposal_model,
    relation_slot_count,
    render_layout_prompt,
    with_bound,
    with_reason,
)
from sema.intake.shapes.base import RelationProfile, SlotBinding, profile_relations
from sema.intake.shapes.base_utils import failure
from sema.intake.structure import BoundEntry, StructureResult
from sema.intake.structure_utils import describe_failure
from sema.intake.verify.base import Failure
from sema.intake.verify.structure import verify_structure
from sema.intake.views import ViewSpec
from sema.intake.views.verify_utils import verify_over
from sema.llm_client import LLMStageError
from sema.log import logger

__all__ = [
    "DEFAULT_MAX_RETRIES",
    "LayoutAttempt",
    "LayoutFinding",
    "ProposalResult",
    "propose_layouts",
]

STAGE = "propose_layouts"
FINDING_KIND = "layout_finding"
NEEDS_VALUES = "needs_values"
MODEL_CALL = "model_call"
PROPOSAL_TEMPLATE = "proposal_template"
PROPOSAL_RELATIONS = "proposal_relations"
PROPOSAL_TARGET = "proposal_covers_target"
VIEW_COLUMNS = "view_columns_named"
VALUES_NAMED = "needs_values_named"


@dataclass(frozen=True)
class LayoutAttempt:
    """One call: its key, the model's answer (None when the call failed), its failures."""

    call_key: str
    proposal: BaseModel | None
    failures: tuple[Failure, ...]


@dataclass(frozen=True)
class LayoutFinding:
    relation: str
    rationale: str


@dataclass(frozen=True)
class ProposalResult:
    structure: StructureResult
    attempts: dict[str, tuple[LayoutAttempt, ...]]
    escalated: tuple[str, ...]
    findings: tuple[LayoutFinding, ...]
    value_requests: tuple[NeedsValues, ...]


@dataclass(frozen=True)
class _Outcome:
    attempt: LayoutAttempt
    entry: BoundEntry | None = None


@dataclass
class _Run:
    land: LandResult
    tools: ProfileTools
    provider: Provider
    calls: ModelCalls
    profiles: dict[str, RelationProfile]
    structure: StructureResult
    attempts: dict[str, tuple[LayoutAttempt, ...]] = field(default_factory=dict)
    escalated: list[str] = field(default_factory=list)
    findings: list[LayoutFinding] = field(default_factory=list)
    value_requests: list[NeedsValues] = field(default_factory=list)


def propose_layouts(
    structure: StructureResult,
    land: LandResult,
    tools: ProfileTools,
    calls: ModelCalls,
    *,
    provider: Provider = "staging",
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ProposalResult:
    profiles = profile_relations(land, tools, provider)
    by_name = {profile.ref.name: profile for profile in profiles}
    run = _Run(land, tools, provider, calls, by_name, structure)
    for relation in structure.proposal_targets:
        bound = {name for entry in run.structure.bound for name in entry.binding.relations}
        if relation in by_name and relation not in bound:
            _propose(run, relation, max_retries)
    made = len(run.attempts)
    logger.info(
        "propose layouts: {} targets asked, {} bound, {} escalated, {} findings, {} value asks",
        made, len(run.structure.bound) - len(structure.bound), len(run.escalated),
        len(run.findings), len(run.value_requests),
    )
    return ProposalResult(
        structure=run.structure,
        attempts=run.attempts,
        escalated=tuple(run.escalated),
        findings=tuple(run.findings),
        value_requests=tuple(run.value_requests),
    )


def _propose(run: _Run, relation: str, max_retries: int) -> None:
    profile = run.profiles[relation]
    group = _group(run, relation)
    values = layout_values(run.tools, profile)
    made: list[LayoutAttempt] = []

    def ask(failed: Sequence[Sequence[str]]) -> _Outcome:
        outcome = _ask(run, profile, group, values, failed)
        made.append(outcome.attempt)
        return outcome

    outcome, passed = retried(
        ask,
        lambda o: _notes(o.attempt.failures),
        max_retries,
        ended=lambda o: o.attempt.proposal is None,
    )
    run.attempts[relation] = tuple(made)
    if not passed:
        run.escalated.append(relation)
        reason = "; ".join(_notes(outcome.attempt.failures))
        run.structure = with_reason(run.structure, relation, reason)
        logger.info(f"layout proposal for {relation} escalated after {len(made)} attempts")
        return
    _settle(run, relation, outcome)


def _settle(run: _Run, relation: str, outcome: _Outcome) -> None:
    answer = outcome.attempt.proposal
    if outcome.entry is not None:
        run.structure = with_bound(run.structure, outcome.entry)
        return
    if answer is None:
        return
    rationale = str(getattr(answer, "reason"))
    if str(getattr(answer, "answer")) == NEEDS_VALUES:
        columns = tuple(getattr(answer, NEEDS_VALUES))
        ref = run.profiles[relation].ref
        if not all(run.tools.values_approved(ref, column) for column in columns):
            run.value_requests.append(NeedsValues(relation=relation, columns=columns))
            return
        asked = ", ".join(columns)
        rationale = f"values were asked again for approved columns {asked}: {rationale}"
    run.findings.append(LayoutFinding(relation=relation, rationale=rationale))
    payload = dict(relation=relation, rationale=rationale, call_key=outcome.attempt.call_key)
    run.tools.ledger.event(STAGE, FINDING_KIND, payload)
    run.structure = with_reason(run.structure, relation, rationale)


def _ask(
    run: _Run,
    profile: RelationProfile,
    group: Sequence[str],
    values: ShownValues,
    failed: Sequence[Sequence[str]],
) -> _Outcome:
    relation = profile.ref.name
    members = [run.profiles[name] for name in group if name in run.profiles]
    prompt = render_layout_prompt(profile, members, values, failed)
    output_model = proposal_model(profile, group)
    key = run.calls.call_key(prompt, output_model)
    try:
        answer = run.calls.call(STAGE, prompt, output_model)
    except LLMStageError as error:
        logger.warning(f"Layout proposal call {key} for {relation} failed: {error}")
        return _Outcome(LayoutAttempt(key, None, (failure(MODEL_CALL, relation),)))
    kind = str(getattr(answer, "answer"))
    if kind == ABSTAIN:
        return _Outcome(LayoutAttempt(key, answer, ()))
    if kind == NEEDS_VALUES:
        named = bool(getattr(answer, NEEDS_VALUES))
        failures = () if named else (failure(VALUES_NAMED, relation),)
        return _Outcome(LayoutAttempt(key, answer, failures))
    entry, found = _verified(run, relation, key, answer)
    return _Outcome(LayoutAttempt(key, answer, tuple(found)), None if found else entry)


def _verified(
    run: _Run, relation: str, key: str, answer: BaseModel
) -> tuple[BoundEntry | None, list[Failure]]:
    """View, then the template's verify, then the Structure verifier over what it spans."""
    template = next((t for t in proposable() if t.ID == getattr(answer, "template")), None)
    if template is None:
        return None, [failure(PROPOSAL_TEMPLATE, relation)]
    binding = binding_of(answer)
    view, failures = _view(relation, answer)
    failures += _shape_failures(relation, answer, binding)
    if failures:
        return None, failures
    profiles = list(run.profiles.values())
    failures = verify_over(template, binding, view, profiles, run.tools)
    if failures:
        return None, failures
    confidence = min(max(float(getattr(answer, "confidence")), 0.0), 1.0)
    entry = BoundEntry(
        binding=binding, confidence=confidence, source="model", call_key=key, view=view
    )
    proposal = with_bound(run.structure, entry)
    verdict = verify_structure(proposal, run.land, run.tools, provider=run.provider)
    spanned = set(binding.relations)
    return entry, [found for found in verdict.failures if found.relation in spanned]


def _view(relation: str, answer: BaseModel) -> tuple[ViewSpec | None, list[Failure]]:
    named = str(getattr(answer, "view"))
    if named == NONE:
        return None, []
    columns = tuple(getattr(answer, "view_columns"))
    try:
        return ViewSpec(view=named, relation=relation, columns=columns), []
    except ValidationError:
        return None, [failure(VIEW_COLUMNS, relation, view=named, columns=len(columns))]


def _shape_failures(relation: str, answer: BaseModel, binding: SlotBinding) -> list[Failure]:
    stated = tuple(dict.fromkeys(getattr(answer, "relations")))
    slots = relation_slot_count(binding.template, binding)
    failures = []
    if len(stated) > slots:
        failures.append(failure(PROPOSAL_RELATIONS, relation, stated=len(stated), slots=slots))
    if relation not in binding.relations:
        failures.append(failure(PROPOSAL_TARGET, relation, template=binding.template))
    return failures


def _group(run: _Run, relation: str) -> tuple[str, ...]:
    return next((g for g in run.structure.groups if relation in g), (relation,))


def _notes(failures: Sequence[Failure]) -> list[str]:
    return list(dict.fromkeys(describe_failure(found) for found in failures))
