"""The open entries Review writes when assembly produces no spec (decision 14).

A run that reaches Review with bindings and no spec names why: each required role settled
``none`` and each role the per-file dictionaries read from differently named columns reopens
as an entry carrying its ``cause``. A section two bindings fill is already open at Structure
with the section in its reason and every member offered, so it needs no entry here. Any other
failure of a run with no open entry names each bound section with the assembler's reason.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import replace

from pydantic import BaseModel

from sema.binding.spec_sections import EntityModelRoles
from sema.intake import lexicon
from sema.intake.eligibility_spec_utils import RolesResult, roles_of
from sema.intake.eligibility_target_utils import Target, role_targets
from sema.intake.eligibility_text_utils import STAGE as ROLES_STAGE
from sema.intake.eligibility_utils import RoleEligibility, scope_of, subject_of
from sema.intake.shapes import entity_model, filled_sections
from sema.intake.shapes.base import SlotBinding
from sema.intake.verify.base import Ambiguity

__all__ = [
    "REQUIRED_ROLE_NONE", "DIVERGENT_ROLE_MAP", "SPEC_UNASSEMBLED", "CauseEntry", "with_causes",
]

REQUIRED_ROLE_NONE = "required_role_none"
DIVERGENT_ROLE_MAP = "divergent_role_map"
SPEC_UNASSEMBLED = "spec_unassembled"


class CauseEntry(Ambiguity):
    """An open entry naming why no spec assembled; the reader prints its ``cause`` last."""

    cause: str


def with_causes(roles: RolesResult, unassembled: str | None = None) -> RolesResult:
    """``roles`` with every entry a cause names reopened, or a new entry where none is listed;
    ``unassembled`` is the assembler's reason, named only where nothing else is open."""
    causes = {entry.subject: entry for entry in _causes(roles)}
    if not causes and unassembled is not None and not _any_open(roles):
        causes = {entry.subject: entry for entry in _unassembled(roles, unassembled)}
    if not causes:
        return roles
    listed = {entry.subject: entry for entry in roles.ambiguities}
    entries = [
        _reopened(listed[subject], entry) if subject in listed else entry
        for subject, entry in causes.items()
    ]
    kept = tuple(entry for entry in roles.ambiguities if entry.subject not in causes)
    return replace(roles, ambiguities=(*kept, *entries))


def _reopened(listed: Ambiguity, cause: CauseEntry) -> CauseEntry:
    opened = listed.model_copy(update=dict(
        choice=None, confidence=None, source=None, call_key=None,
        reason=cause.reason or listed.reason,
    ))
    return CauseEntry(**opened.model_dump(), cause=cause.cause)


def _causes(roles: RolesResult) -> Iterator[CauseEntry]:
    bound = [entry.binding for entry in roles.structure.bound]
    for binding in bound:
        for target in role_targets(binding):
            yield from _required_none(target, roles)
    yield from _divergent([b for b in bound if _per_file(b)], roles)


def _any_open(roles: RolesResult) -> bool:
    listed = (*roles.structure.ambiguities, *roles.ambiguities)
    return any(entry.choice is None and entry.source is None for entry in listed)


def _unassembled(roles: RolesResult, reason: str) -> Iterator[CauseEntry]:
    for entry in roles.structure.bound:
        relation = entry.binding.relations[0]
        for section in filled_sections(entry.binding) or (entry.binding.template,):
            yield CauseEntry(
                stage=ROLES_STAGE, subject=f"{relation}.{section}", members=(), reason=reason,
                cause=SPEC_UNASSEMBLED,
            )


def _per_file(binding: SlotBinding) -> bool:
    return (
        binding.template == entity_model.ID
        and binding.slots.get(entity_model.LAYOUT) == entity_model.PER_FILE
    )


def _required_none(target: Target, roles: RolesResult) -> Iterator[CauseEntry]:
    """Settled entries with no column for a role every model of the target requires."""
    required = set.intersection(*(_required(model) for model in target.models))
    stated = roles.stated.get(scope_of(target.relation, target.section), {})
    for eligibility in roles.eligibilities:
        if (
            eligibility.relation == target.relation
            and eligibility.section == target.section
            and eligibility.subject is None
            and eligibility.role in required
            and eligibility.role not in stated
            and eligibility.source is not None
            and eligibility.column is None
        ):
            yield _entry(eligibility, REQUIRED_ROLE_NONE)


def _required(model: type[BaseModel]) -> set[str]:
    return {name for name, field in model.model_fields.items() if field.is_required()}


def _divergent(bindings: Sequence[SlotBinding], roles: RolesResult) -> Iterator[CauseEntry]:
    """Each role the dictionaries' role maps disagree on, once per dictionary."""
    dictionaries = [b.slots[entity_model.DICTIONARY] for b in bindings]
    maps = {d: _role_map(d, roles) for d in dictionaries}
    if len(maps) < 2 or any(found is None for found in maps.values()):
        return
    dumped = {d: found.model_dump(mode="json") for d, found in maps.items() if found}
    for role in EntityModelRoles.model_fields:
        columns = {d: json.dumps(values[role]) for d, values in dumped.items()}
        if len(set(columns.values())) < 2:
            continue
        read = ", ".join(f"{d} reads {column}" for d, column in columns.items())
        for dictionary in dictionaries:
            yield _role_entry(roles, dictionary, role, f"the dictionaries differ: {read}")


def _role_map(dictionary: str, roles: RolesResult) -> EntityModelRoles | None:
    notes = {lexicon.NOTES_ROLE: list(roles.notes.get(dictionary, ()))}
    return roles_of(dictionary, EntityModelRoles, roles, notes)


def _role_entry(roles: RolesResult, relation: str, role: str, reason: str) -> CauseEntry:
    for eligibility in roles.eligibilities:
        if eligibility.relation == relation and eligibility.role == role and (
            eligibility.subject is None and eligibility.section is None
        ):
            return _entry(eligibility, DIVERGENT_ROLE_MAP, reason)
    subject = f"{relation}.{role}"
    return CauseEntry(
        stage=ROLES_STAGE, subject=subject, members=(), reason=reason, cause=DIVERGENT_ROLE_MAP
    )


def _entry(eligibility: RoleEligibility, cause: str, reason: str | None = None) -> CauseEntry:
    return CauseEntry(
        stage=ROLES_STAGE,
        subject=subject_of(eligibility),
        members=eligibility.members,
        exclusions=eligibility.exclusions,
        reason=reason,
        cause=cause,
    )
