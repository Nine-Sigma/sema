"""Review decisions: a reviewer's choices settle open entries as human choices, no model call.

A decisions file is a YAML list of ``{stage, subject, choice}``. Each decision is checked
against the ambiguity list Review presented; ``choice`` is a member of its entry or
``IGNORE``. A structure decision binds the chosen member's binding, or ignores the relation
with reason ``human``, and the entry leaves the ambiguity list. Role decisions enter Roles as
stated choices; ``IGNORE`` states ``none``. Every decided binding is still verified.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from sema.binding.spec_sections import Provider
from sema.intake import lexicon
from sema.intake.eligibility_text_utils import STAGE as ROLES_STAGE
from sema.intake.eligibility_utils import HUMAN, StatedChoice
from sema.intake.landing import LandResult
from sema.intake.profile import ProfileTools
from sema.intake.resolve_utils import member_bindings, open_entries
from sema.intake.shapes import ignore
from sema.intake.shapes.base import SlotBinding, profile_relations
from sema.intake.shapes.precedence import STAGE as STRUCTURE_STAGE
from sema.intake.structure import BindingSource, BoundEntry, StructureResult
from sema.intake.verify.base import Ambiguity
from sema.log import logger

__all__ = [
    "IGNORE",
    "Decision",
    "DecisionRejectedError",
    "apply_structure_decisions",
    "load_decisions",
    "stated_choices",
    "validate_decisions",
]

IGNORE = "ignore"
HUMAN_CONFIDENCE = 1.0
HUMAN_BINDING: BindingSource = "human"
NOT_LISTED = "no entry Review presented has this stage and subject"
NOT_A_MEMBER = "the choice is neither a member of the entry nor ignore"
NO_BINDING = "the choice names no binding of an open structure entry"


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str
    subject: str
    choice: str


class DecisionRejectedError(ValueError):
    """A decision that settles nothing Review presented; nothing re-runs."""

    def __init__(self, decision: Decision, members: Sequence[str], problem: str) -> None:
        self.subject, self.choice, self.members = decision.subject, decision.choice, tuple(members)
        listed = ", ".join(self.members) or "none"
        super().__init__(
            f"decision {self.choice!r} for {decision.stage} entry {self.subject} is rejected:"
            f" {problem}; members: {listed}"
        )


def load_decisions(path: Path) -> tuple[Decision, ...]:
    rows = yaml.safe_load(path.read_text()) or []
    if not isinstance(rows, list):
        raise ValueError(f"decisions file {path} is not a list of {{stage, subject, choice}}")
    return tuple(Decision.model_validate(row) for row in rows)


def validate_decisions(
    decisions: Iterable[Decision], ambiguities: Sequence[Ambiguity]
) -> tuple[Decision, ...]:
    """``ambiguities`` is the list Review presented, as the caller read it from the ledger."""
    listed = {(entry.stage, entry.subject): entry for entry in ambiguities}
    checked = tuple(decisions)
    for decision in checked:
        entry = listed.get((decision.stage, decision.subject))
        if entry is None:
            raise DecisionRejectedError(decision, (), NOT_LISTED)
        if decision.choice != IGNORE and decision.choice not in entry.members:
            raise DecisionRejectedError(decision, entry.members, NOT_A_MEMBER)
    return checked


def apply_structure_decisions(
    structure: StructureResult,
    decisions: Iterable[Decision],
    land: LandResult,
    tools: ProfileTools,
    *,
    provider: Provider = "staging",
) -> StructureResult:
    chosen = {d.subject: d for d in decisions if d.stage == STRUCTURE_STAGE}
    if not chosen:
        return structure
    profiles = profile_relations(land, tools, provider)
    by_name = {profile.ref.name: profile for profile in profiles}
    pending = {
        entry.ambiguity.subject: entry
        for entry in open_entries(structure, member_bindings(structure, profiles, tools))
    }
    bound: dict[str, BoundEntry] = {}
    ignored: list[SlotBinding] = []
    for subject, decision in chosen.items():
        if decision.choice == IGNORE and subject in by_name:
            ignored.append(ignore.decided(by_name[subject], tools))
            continue
        entry = pending.get(subject)
        if entry is None or decision.choice not in entry.bindings:
            members = entry.ambiguity.members if entry is not None else ()
            raise DecisionRejectedError(decision, members, NO_BINDING)
        binding = entry.bindings[decision.choice]
        bound.setdefault(decision.choice, _human(binding))
    dropped = {name for binding in ignored for name in binding.relations}
    logger.info(
        "structure decisions: {} bound, {} ignored as human", len(bound), len(ignored)
    )
    return structure.model_copy(update=dict(
        bound=(*structure.bound, *bound.values()),
        ignored=(*structure.ignored, *ignored),
        ambiguities=tuple(a for a in structure.ambiguities if not _decided(a, chosen)),
        proposal_targets=tuple(t for t in structure.proposal_targets if t not in dropped),
    ))


def stated_choices(decisions: Iterable[Decision]) -> dict[str, StatedChoice]:
    """Role, reference, and section decisions by subject, for ``run_roles(choices=...)``."""
    return {
        d.subject: StatedChoice(
            choice=lexicon.NO_REFERENCE if d.choice == IGNORE else d.choice,
            source=HUMAN,
            confidence=HUMAN_CONFIDENCE,
        )
        for d in decisions
        if d.stage == ROLES_STAGE
    }


def _human(binding: SlotBinding) -> BoundEntry:
    return BoundEntry(binding=binding, confidence=HUMAN_CONFIDENCE, source=HUMAN_BINDING)


def _decided(ambiguity: Ambiguity, chosen: dict[str, Decision]) -> bool:
    return ambiguity.stage == STRUCTURE_STAGE and ambiguity.subject in chosen
