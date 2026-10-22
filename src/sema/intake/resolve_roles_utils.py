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
from sema.intake.eligibility_text_utils import STAGE
from sema.intake import lexicon
from sema.intake.eligibility_utils import (
    MemoryKey,
    RoleEligibility,
    claimed_columns,
    scope_of,
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
from sema.intake.output_models_utils import Choice
from sema.intake.resolve_utils import (
    NOTHING,
    SchemaText,
    bullets,
    history,
    render_relations,
    render_template,
)
from sema.intake.shapes.base import RelationProfile
from sema.intake.verify.base import Ambiguity

ROLES_PROMPT = "prompts/roles_v1.md"
NOT_IN_NAME = re.compile(r"\W+")


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


def column_asks(roles: RolesResult, by_name: Mapping[str, RelationProfile]) -> list[RoleAsk]:
    """The first relation of each signature with open column roles; later ones are kin."""
    found = _open(roles, per_property=False)
    claims = _claims(roles)
    keys = _signature_keys(roles, by_name)
    asks: dict[MemoryKey | str, RoleAsk] = {}
    for relation, entries in found.items():
        key = keys.get(relation, relation)
        if key in asks:
            ask = asks[key]
            asks[key] = RoleAsk(ask.relation, ask.entries, False, (*ask.kin, relation))
            continue
        asks[key] = RoleAsk(relation, _columns(entries, claims), False)
    return list(asks.values())


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


def _offered(entry: RoleEligibility, ambiguity: Ambiguity) -> tuple[str, ...]:
    return tuple(member for member in entry.members if member not in ambiguity.exclusions)


def _claims(roles: RolesResult) -> dict[str, dict[str, str]]:
    """Columns code or the lexicon settled, by role, per side of each relation."""
    sides: dict[str, list[RoleEligibility]] = {}
    for entry in roles.eligibilities:
        if entry.subject is None:
            sides.setdefault(scope_of(entry.relation, entry.section), []).append(entry)
    return {side: claimed_columns(entries) for side, entries in sides.items()}


def _unclaimed(
    entry: RoleEligibility, ambiguity: Ambiguity, claims: Mapping[str, Mapping[str, str]]
) -> Ambiguity:
    """The entry with every column another role on its side claims listed as excluded."""
    side = claims.get(scope_of(entry.relation, entry.section), {})
    claimed = {
        column: role
        for column, role in side.items()
        if column in entry.members and column not in ambiguity.exclusions
        and role != entry.role and not lexicon.coincide(role, entry.role)
    }
    if not claimed:
        return ambiguity
    return ambiguity.model_copy(update={"exclusions": {**ambiguity.exclusions, **claimed}})


def _columns(
    entries: Sequence[tuple[RoleEligibility, Ambiguity]],
    claims: Mapping[str, Mapping[str, str]],
) -> tuple[OpenRole, ...]:
    """A role open on both sides of one relation names each side's field apart."""
    counts = Counter(entry.role for entry, _ in entries)
    return tuple(
        _column(e, _unclaimed(e, a, claims), repeated=counts[e.role] > 1) for e, a in entries
    )


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


def render_roles_prompt(
    ask: RoleAsk,
    claims: Sequence[RoleEligibility],
    relations: Sequence[RelationProfile],
    texts: SchemaText,
    failed: Sequence[Sequence[str]],
) -> str:
    sections = {
        "in_force": bullets(f"`{e.role}`: `{e.choice}` ({e.source})" for e in claims),
        "failures": history(failed),
        "relations": render_relations(relations, texts),
        "entries": "\n\n".join(_entry(index, entry) for index, entry in enumerate(ask.entries)),
    }
    return render_template(ROLES_PROMPT, sections)


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


def chosen_notes(entries: Iterable[OpenRole], choices: Mapping[str, str]) -> str:
    return ", ".join(
        f"`{entry.label}` chose `{choices[entry.subject]}`"
        for entry in entries
        if entry.subject in choices
    )
