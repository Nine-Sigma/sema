"""Resolve structure: the model settles open structure entries among the members code proved.

Entries go out in batches of at most ``MAX_STRUCTURE_BATCH``, one call each through the
replayed, capped model-call path. A choice is written onto its entry with ``source: model``,
the confidence, and the call key, and its binding is bound with ``source: model``. The
Structure verifier re-runs on every answer; a batch whose choices fail is re-asked with the
failed checks and the bindings in force, at most ``max_retries`` times, and then only the
choices spanning a failing relation are left open, each with the failure as its reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sema.binding.spec_sections import Provider
from sema.intake.landing import LandResult
from sema.intake.model_calls import ModelCalls
from sema.intake.output_models import (
    MAX_STRUCTURE_BATCH,
    entry_choices,
    structure_batch_model,
)
from sema.intake.profile import ProfileTools
from sema.intake.resolve_utils import (
    OpenEntry,
    Settlement,
    failure_notes,
    member_bindings,
    open_entries,
    open_with,
    render_prompt,
    reopened,
    settle_choices,
    text_columns,
)
from sema.intake.shapes.base import RelationProfile, profile_relations
from sema.intake.shapes.precedence import STAGE
from sema.intake.structure import BoundEntry, StructureResult
from sema.intake.verify.base import Ambiguity, Failure
from sema.intake.verify.structure import verify_structure
from sema.llm_client import LLMStageError
from sema.log import logger

__all__ = ["DEFAULT_MAX_RETRIES", "resolve_structure"]

DEFAULT_MAX_RETRIES = 2
MISALIGNED = "the answer's confidences or reasons do not list one item per entry"


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
    result = run.proposal()
    settled = sum(1 for entry in pending if run.ambiguities[entry.position].choice is not None)
    logger.info(
        "resolve structure: {} of {} open entries settled by the model, {} bound",
        settled, len(pending), len(result.bound) - len(structure.bound),
    )
    return result


def _settle(run: _Run, batch: Sequence[OpenEntry], max_retries: int) -> None:
    failed: list[list[str]] = []
    attempt = _Attempt(None, ())
    for _ in range(max_retries + 1):
        attempt = _ask(run, batch, failed)
        if attempt.settlement is not None and not attempt.failures:
            run.apply(attempt.settlement)
            return
        failed.append(_notes(batch, attempt))
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
        return _Attempt(None, (), f"the model call failed: {error}")
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
        problem = attempt.problem or MISALIGNED
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
    return render_prompt(batch, in_force, relations, _schema_text(run, relations), failed)


def _schema_text(
    run: _Run, relations: Sequence[RelationProfile]
) -> dict[str, list[tuple[object, ...]]]:
    """Text of the bound schema relations among ``relations``, only when the run may read it."""
    readable = run.tools.structure.schema_relations if run.tools.structure is not None else ()
    return {
        profile.ref.name: run.tools.read_schema_text(profile.ref, text_columns(profile))
        for profile in relations
        if profile.ref.name in readable and text_columns(profile)
    }
