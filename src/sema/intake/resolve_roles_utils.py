"""Role resolution helpers: open entries grouped into calls, their output models, the prompt.

Column roles go out one call per relation signature, so a relation sharing the signature
inherits the verified answer through memory. Per-property reference entries differ per
relation, so they go out one call per relation and are never inherited.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from sema.intake.eligibility import role_targets
from sema.intake.eligibility_spec_utils import RolesResult
from sema.intake.eligibility_text_utils import STAGE
from sema.intake.eligibility_utils import MemoryKey, RoleEligibility, subject_of
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


@dataclass(frozen=True)
class OpenRole:
    """``label`` is the role of a column entry or the subject of a reference entry."""

    subject: str
    label: str
    field: str
    members: tuple[str, ...]
    ambiguity: Ambiguity


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
        return roles_model({entry.label: entry.members for entry in self.entries})

    def choices(self, answer: BaseModel) -> list[Choice] | None:
        if self.references:
            return reference_choices(answer, len(self.entries))
        return role_choices(answer, [entry.label for entry in self.entries])


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
        asks[key] = RoleAsk(relation, tuple(_column(e, a) for e, a in entries), False)
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


def _column(entry: RoleEligibility, ambiguity: Ambiguity) -> OpenRole:
    offered = _offered(entry, ambiguity)
    return OpenRole(ambiguity.subject, entry.role, role_field(entry.role), offered, ambiguity)


def _reference(index: int, entry: RoleEligibility, ambiguity: Ambiguity) -> OpenRole:
    field = reference_field(index)
    return OpenRole(ambiguity.subject, ambiguity.subject, field, entry.members, ambiguity)


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
