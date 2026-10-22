"""Role resolution helpers: open entries grouped into calls, their output models, the prompt.

Column roles go out one call per relation signature, so a relation sharing the signature
inherits the verified answer through memory. Per-property reference entries differ per
relation, so they go out one call per relation and are never inherited.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from sema.intake.eligibility import role_targets
from sema.intake.eligibility_spec_utils import RolesResult
from sema.intake.eligibility_text_utils import (
    SECTION_MEMBERS,
    SECTION_SUBJECT,
    STAGE,
    section_subject,
)
from sema.intake.eligibility_utils import (
    MemoryKey,
    RoleEligibility,
    StatedChoice,
    subject_of,
)
from sema.intake.output_models import (
    MAX_REFERENCE_BATCH,
    reference_choices,
    reference_field,
    references_model,
    role_choices,
    role_field,
    roles_model,
)
from sema.intake.output_models_utils import NONE, Choice
from sema.intake.resolve_budget_utils import STATEMENTS, PromptEvidence, fitted
from sema.intake.resolve_utils import (
    NOTHING,
    bullets,
    history,
    render_template,
)
from sema.intake.shapes import concept_list
from sema.intake.shapes.base import RelationProfile
from sema.intake.verify.base import Ambiguity

ROLES_PROMPT = "prompts/roles_v1.md"
SECTION_QUESTION = "asks: the spec section this relation's concept list feeds"
NOT_IN_NAME = re.compile(r"\W+")
UNASKED_CLAIMED = "not asked: every member is claimed ({})"
UNASKED_KIN = "not asked: it shares the signature of {}, whose call left it open"
UNASKED_SECTION = "not asked: the section choice on {} opened it after its ask; left for Review"


@dataclass(frozen=True)
class OpenRole:
    """``label`` is the role of a column entry or the subject of a reference entry;
    ``name`` keys the entry's output field, qualified by section when a role repeats."""

    subject: str
    label: str
    field: str
    members: tuple[str, ...]
    ambiguity: Ambiguity
    name: str


@dataclass(frozen=True)
class RoleAsk:
    """One call: open entries of ``relation``; ``kin`` share its signature and inherit."""

    relation: str
    entries: tuple[OpenRole, ...]
    references: bool
    kin: tuple[str, ...] = ()

    def output_model(self) -> type[BaseModel]:
        if self.references:
            return references_model([entry.members for entry in self.entries])
        return roles_model({entry.name: entry.members for entry in self.entries})

    def choices(self, answer: BaseModel) -> list[Choice] | None:
        if self.references:
            return reference_choices(answer, len(self.entries))
        return role_choices(answer, [entry.name for entry in self.entries])

    def columns(self, stated: Mapping[str, StatedChoice]) -> set[str]:
        """The columns a stated answer binds; a section choice binds none."""
        if self.references:
            return set()
        return {
            stated[entry.subject].choice
            for entry in self.entries
            if entry.subject in stated and entry.label != SECTION_SUBJECT
        }


def column_asks(roles: RolesResult, by_name: Mapping[str, RelationProfile]) -> list[RoleAsk]:
    """The first relation of each signature with open column roles; later ones are kin."""
    found = _open(roles, per_property=False)
    keys = _signature_keys(roles, by_name)
    asks: dict[MemoryKey | str, RoleAsk] = {}
    for relation, entries in found.items():
        key = keys.get(relation, relation)
        if key in asks:
            ask = asks[key]
            asks[key] = RoleAsk(ask.relation, ask.entries, False, (*ask.kin, relation))
            continue
        asks[key] = RoleAsk(relation, _columns(entries), False)
    return list(asks.values())


def unasked_reason(
    ambiguity: Ambiguity, asks: Sequence[RoleAsk], accepted: Mapping[str, StatedChoice]
) -> str | None:
    """Why an open entry no call offered stays open; ``None`` when no path explains it."""
    members = ambiguity.members
    if members and all(member in ambiguity.exclusions for member in members):
        claims = ", ".join(f"{m} by {ambiguity.exclusions[m]}" for m in members)
        return UNASKED_CLAIMED.format(claims)
    subject = ambiguity.subject
    for ask in asks:
        if any(subject.startswith(f"{kin}.") for kin in ask.kin):
            return UNASKED_KIN.format(ask.relation)
        sectioned = section_subject(ask.relation) in accepted
        if sectioned and subject.startswith(f"{ask.relation}."):
            return UNASKED_SECTION.format(ask.relation)
    return None


def reference_asks(roles: RolesResult) -> list[RoleAsk]:
    asks = []
    for relation, entries in _open(roles, per_property=True).items():
        for start in range(0, len(entries), MAX_REFERENCE_BATCH):
            batch = entries[start : start + MAX_REFERENCE_BATCH]
            listed = tuple(_reference(i, e, a) for i, (e, a) in enumerate(batch))
            asks.append(RoleAsk(relation, listed, True))
    return asks


def _open(
    roles: RolesResult, *, per_property: bool
) -> dict[str, list[tuple[RoleEligibility, Ambiguity]]]:
    """Open entries with a choice to make, by relation in Roles order."""
    eligible = {
        subject_of(e): e for e in roles.eligibilities if (e.subject is not None) == per_property
    }
    if not per_property:
        eligible |= _open_sections(roles)
    # A reference entry always offers `none`, so one member leaves nothing to choose.
    fewest = 2 if per_property else 1
    found: dict[str, list[tuple[RoleEligibility, Ambiguity]]] = {}
    for ambiguity in roles.ambiguities:
        entry = eligible.get(ambiguity.subject)
        settled = ambiguity.choice is not None or ambiguity.source is not None
        if ambiguity.stage != STAGE or settled or entry is None:
            continue
        if len(_offered(entry, ambiguity)) >= fewest:
            found.setdefault(entry.relation, []).append((entry, ambiguity))
    return found


def _open_sections(roles: RolesResult) -> dict[str, RoleEligibility]:
    """A concept list with no section yet is asked its section with its column roles."""
    relations = [
        entry.binding.relations[0]
        for entry in roles.structure.bound
        if entry.binding.template == concept_list.ID and entry.binding.section is None
    ]
    return {
        section_subject(relation): RoleEligibility(
            relation=relation, role=SECTION_SUBJECT, members=SECTION_MEMBERS
        )
        for relation in relations
    }


def _offered(entry: RoleEligibility, ambiguity: Ambiguity) -> tuple[str, ...]:
    return tuple(member for member in entry.members if member not in ambiguity.exclusions)


def _columns(entries: Sequence[tuple[RoleEligibility, Ambiguity]]) -> tuple[OpenRole, ...]:
    """A role open on both sides of one relation names each side's field apart."""
    counts = Counter(entry.role for entry, _ in entries)
    return tuple(_column(e, a, repeated=counts[e.role] > 1) for e, a in entries)


def _column(entry: RoleEligibility, ambiguity: Ambiguity, *, repeated: bool) -> OpenRole:
    offered = _offered(entry, ambiguity)
    name = entry.role
    if repeated and entry.section is not None:
        name = f"{NOT_IN_NAME.sub('_', entry.section)}_{entry.role}"
    return OpenRole(ambiguity.subject, entry.role, role_field(name), offered, ambiguity, name)


def _reference(index: int, entry: RoleEligibility, ambiguity: Ambiguity) -> OpenRole:
    field = reference_field(index)
    return OpenRole(ambiguity.subject, ambiguity.subject, field, entry.members, ambiguity, field)


def _signature_keys(
    roles: RolesResult, by_name: Mapping[str, RelationProfile]
) -> dict[str, MemoryKey]:
    """The key Roles memory files each relation's role map under."""
    return {
        target.relation: (target.template, by_name[target.relation].signature)
        for entry in roles.structure.bound
        for target in role_targets(entry.binding)
        if target.relation in by_name
    }


def spanned(roles: RolesResult, relation: str) -> list[str]:
    """``relation`` and every relation bound with it, such as a dictionary's header."""
    names = [relation]
    for entry in roles.structure.bound:
        if relation in entry.binding.relations:
            names += entry.binding.relations
    return list(dict.fromkeys(names))


def in_force(roles: RolesResult, relation: str) -> list[RoleEligibility]:
    return [
        e for e in roles.eligibilities
        if e.relation == relation and e.subject is None and e.choice is not None
    ]


def fitted_roles_prompt(
    ask: RoleAsk,
    claims: Sequence[RoleEligibility],
    evidence: PromptEvidence,
    failed: Sequence[Sequence[str]],
) -> str:
    """The Roles prompt, compacted only as far as the budget needs."""
    held = {e.choice for e in claims if e.choice is not None}
    kept = {ask.relation: {m for e in ask.entries for m in e.members} | held}
    asked = next((p for p in evidence.relations if p.ref.name == ask.relation), None)
    columns = tuple(c.name for c in asked.columns) if asked is not None else ()

    def render(level: int) -> str:
        stated = columns if level >= STATEMENTS else None
        return render_roles_prompt(ask, claims, evidence.render(level, kept), failed, stated)

    return fitted(render, ask.output_model())


def render_roles_prompt(
    ask: RoleAsk,
    claims: Sequence[RoleEligibility],
    relations: str,
    failed: Sequence[Sequence[str]],
    columns: Sequence[str] | None = None,
) -> str:
    """With ``columns``, an entry offering each of them but its exclusions says so."""
    entries = (_entry(i, entry, ask.relation, columns) for i, entry in enumerate(ask.entries))
    sections = {
        "in_force": bullets(f"`{e.role}`: `{e.choice}` ({e.source})" for e in claims),
        "failures": history(failed),
        "relations": relations,
        "entries": "\n\n".join(entries),
    }
    return render_template(ROLES_PROMPT, sections)


def _entry(index: int, entry: OpenRole, relation: str, columns: Sequence[str] | None) -> str:
    ambiguity = entry.ambiguity
    exclusions = [f"{name}: {why}" for name, why in sorted(ambiguity.exclusions.items())]
    question = [SECTION_QUESTION] if entry.label == SECTION_SUBJECT else []
    return "\n".join(
        [
            f"### entry {index}: {ambiguity.subject}",
            *question,
            f"field: `{entry.field}`",
            *_members(entry, relation, columns),
            f"hints: {', '.join(ambiguity.hints) or NOTHING}",
            f"exclusions: {'; '.join(exclusions) or NOTHING}",
        ]
    )


def _members(entry: OpenRole, relation: str, columns: Sequence[str] | None) -> list[str]:
    offered = [member for member in entry.members if member != NONE]
    remaining = [c for c in columns or () if c not in entry.ambiguity.exclusions]
    if not remaining or sorted(offered) != sorted(remaining):
        return ["members:", bullets(f"`{member}`" for member in entry.members)]
    also = f", and `{NONE}`" if NONE in entry.members else ""
    return [f"members: every column of `{relation}` listed above except the exclusions{also}"]


def chosen_notes(entries: Iterable[OpenRole], choices: Mapping[str, str]) -> str:
    return ", ".join(
        f"`{entry.label}` chose `{choices[entry.subject]}`"
        for entry in entries
        if entry.subject in choices
    )
