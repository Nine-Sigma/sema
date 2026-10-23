"""Role resolution helpers: the asks, their output models and prompts, and their answers.

Column roles go out as one role map per relation signature (S5 decision 1): one field per
role of every role model the relation's bindings use, each naming any column or ``none``.
A relation sharing the signature inherits the verified map through memory. Per-property
reference entries differ per relation, so they go out one call per relation, over the
members code proved, and are never inherited.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, create_model

from sema.intake import lexicon
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
    fact_of,
    scope_of,
    subject_of,
)
from sema.intake.output_models import (
    MAX_REFERENCE_BATCH,
    NOT_IN_NAME,
    reference_choices,
    reference_field,
    references_model,
    role_field,
    role_map_choices,
    role_map_fields,
    role_map_model,
)
from sema.intake.output_models_utils import NONE, Choice
from sema.intake.resolve_prompt_utils import (
    SCHEMA_TEXT_SAMPLE_ROWS as SCHEMA_TEXT_SAMPLE_ROWS,
)
from sema.intake.resolve_prompt_utils import (
    RoleEvidence,
    claim_lines,
    field_lines,
    fitted,
    render_roles_prompt,
)
from sema.intake.resolve_utils import NOTHING, bullets
from sema.intake.shapes import concept_list
from sema.intake.shapes.base import RelationProfile
from sema.intake.verify.base import Ambiguity

SECTION_QUESTION = "the spec section this relation's concept list feeds: {}"
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
class MapSection:
    """One role model of a role map; ``key`` qualifies a repeated role's field and
    ``section`` qualifies its subject."""

    key: str
    section: str | None
    model: type[BaseModel]


@dataclass(frozen=True)
class MapField:
    field: str
    role: str
    subject: str


@dataclass(frozen=True)
class RoleAsk:
    """One call: open entries of ``relation``; ``kin`` share its signature and inherit.
    A column ask carries its role map's ``sections``."""

    relation: str
    entries: tuple[OpenRole, ...]
    references: bool
    kin: tuple[str, ...] = ()
    sections: tuple[MapSection, ...] = ()

    def output_model(self) -> type[BaseModel]:
        if self.references:
            return references_model([entry.members for entry in self.entries])
        return role_map_model({section.key: section.model for section in self.sections})

    def fields(self) -> list[MapField]:
        sections = {section.key: section.section for section in self.sections}
        models = {section.key: section.model for section in self.sections}
        return [
            MapField(name, role, f"{scope_of(self.relation, sections[key])}.{role}")
            for name, key, role in role_map_fields(models)
        ]

    def choices(self, answer: BaseModel) -> list[Choice] | None:
        if self.references:
            return reference_choices(answer, len(self.entries))
        return role_map_choices(answer, [found.field for found in self.fields()])

    def columns(self, stated: Mapping[str, StatedChoice]) -> set[str]:
        """The columns a stated answer binds; a section choice binds none."""
        if self.references:
            return set()
        section = section_subject(self.relation)
        return {choice.choice for subject, choice in stated.items() if subject != section}


def column_asks(roles: RolesResult, by_name: Mapping[str, RelationProfile]) -> list[RoleAsk]:
    """The first relation of each signature with open column roles; later ones are kin."""
    found = _open(roles, per_property=False)
    keys = _signature_keys(roles, by_name)
    asks: dict[MemoryKey | str, RoleAsk] = {}
    for relation, entries in found.items():
        key = keys.get(relation, relation)
        if key in asks:
            ask = asks[key]
            asks[key] = RoleAsk(
                ask.relation, ask.entries, False, (*ask.kin, relation), ask.sections
            )
            continue
        asks[key] = RoleAsk(relation, _columns(entries), False, (), _map_sections(roles, relation))
    return list(asks.values())


def _map_sections(roles: RolesResult, relation: str) -> tuple[MapSection, ...]:
    found: dict[str, MapSection] = {}
    for entry in roles.structure.bound:
        for target in role_targets(entry.binding):
            if target.relation != relation:
                continue
            model = target.models[0] if len(target.models) == 1 else _joined(target.models)
            key = target.section or entry.binding.section or model.__name__
            found.setdefault(key, MapSection(key, target.section, model))
    return tuple(found.values())


def _joined(models: Sequence[type[BaseModel]]) -> type[BaseModel]:
    """The fields of every model a section-less concept list may take, and its section."""
    names = dict.fromkeys(name for model in models for name in model.model_fields)
    fields: dict[str, object] = {name: (str | None, None) for name in names}
    fields[SECTION_SUBJECT] = (str | None, None)
    joined: type[BaseModel] = create_model(
        "".join(model.__name__ for model in models), **fields  # type: ignore[call-overload]
    )
    return joined


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
        eligible |= open_sections(roles)
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


def open_sections(roles: RolesResult) -> dict[str, RoleEligibility]:
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


def map_entries(roles: RolesResult, relation: str) -> dict[str, RoleEligibility]:
    """The column-role and open section entries of ``relation`` by subject."""
    found = {
        subject_of(e): e
        for e in roles.eligibilities
        if e.relation == relation and e.subject is None
    }
    sections = open_sections(roles)
    return found | {s: e for s, e in sections.items() if e.relation == relation}


def fitted_roles_prompt(
    ask: RoleAsk,
    roles: RolesResult,
    evidence: RoleEvidence,
    failed: Sequence[Sequence[str]],
    *,
    shuffled: bool = False,
) -> str:
    """The Roles prompt, compacted only as far as the budget needs; ``shuffled`` is the
    second ask of an agreement pair."""
    claims = claim_lines(in_force(roles, ask.relation))
    if ask.references:
        fields = "\n\n".join(_entry(index, entry) for index, entry in enumerate(ask.entries))
    else:
        width = evidence.width(ask.relation)
        fields = _map_lines(ask, map_entries(roles, ask.relation), width)
    return fitted(
        lambda compaction: render_roles_prompt(
            ask.relation, claims, evidence.render(compaction, shuffled=shuffled), fields, failed
        ),
        ask.output_model(),
    )


def _map_lines(ask: RoleAsk, entries: Mapping[str, RoleEligibility], width: int) -> str:
    described = {
        found.field: _described(found, entries.get(found.subject), width)
        for found in ask.fields()
    }
    return field_lines(described)


def _described(found: MapField, entry: RoleEligibility | None, width: int) -> str:
    """``width`` is the relation's column count: a fact every column meets lists none."""
    if found.role == SECTION_SUBJECT:
        return SECTION_QUESTION.format(", ".join(f"`{m}`" for m in SECTION_MEMBERS))
    known = next((e for e in lexicon.ROLES if e.name == found.role), None)
    parts = [known.description if known else found.role]
    if entry is None:
        return "; ".join(parts)
    parts.append("optional" if lexicon.NO_REFERENCE in entry.members else "required")
    if (fact := fact_of(found.role)) is not None:
        columns = [m for m in entry.members if m != NONE]
        met = ", ".join(f"`{m}`" for m in columns) or NOTHING
        parts.append(f"data fact {fact}" + (f", met by {met}" if len(columns) < width else ""))
    if entry.choice is not None:
        parts.append(f"claim in force: `{entry.choice}` ({entry.source})")
    return "; ".join(parts)


def _entry(index: int, entry: OpenRole) -> str:
    ambiguity = entry.ambiguity
    exclusions = [f"{name}: {why}" for name, why in sorted(ambiguity.exclusions.items())]
    return "\n".join(
        [
            f"### entry {index}: {ambiguity.subject}",
            f"field: `{entry.field}`",
            "members:",
            bullets(f"`{member}`" for member in entry.members),
            f"hints: {', '.join(ambiguity.hints) or NOTHING}",
            f"exclusions: {'; '.join(exclusions) or NOTHING}",
        ]
    )


def chosen_notes(roles: Mapping[str, str], choices: Mapping[str, str]) -> str:
    """``roles`` names each subject's role; ``choices`` the columns chosen for them."""
    return ", ".join(
        f"`{roles.get(subject, subject)}` chose `{column}`" for subject, column in choices.items()
    )


def map_roles(ask: RoleAsk) -> dict[str, str]:
    if ask.references:
        return {entry.subject: entry.label for entry in ask.entries}
    return {found.subject: found.role for found in ask.fields()}


def ranked_first(members: Iterable[str], first: Sequence[str]) -> tuple[str, ...]:
    return (*first, *(member for member in members if member not in first))
