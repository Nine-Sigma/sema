"""Roles verifier helpers: a relation's chosen roles as a bound section, and failure sources.

A relation's chosen roles are re-proved with the same binding and value checks a spec
section gets, so a model answer or a human decision can be verified before any spec
assembles. Every role-check failure names the source of the entries that chose its column.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from typing import Any

from pydantic import BaseModel, create_model

from sema.binding.spec_sections import CodeListRoles, ConceptRoles, StatusRole, ValueSetRoles
from sema.intake import lexicon
from sema.intake.eligibility import Target, role_targets
from sema.intake.eligibility_spec_utils import VALUE_SETS, RolesResult
from sema.intake.eligibility_text_utils import CODE_LISTS, CONCEPTS
from sema.intake.eligibility_utils import RoleEligibility
from sema.intake.structure_hint_utils import KEY_ROLE, SECTION_SEPARATOR
from sema.intake.verify.base import Failure
from sema.intake.verify.roles_utils import BoundSection

CHOSEN_ROLES = "ChosenRoles"
SPEC_SOURCE = "spec"
SECTION_OF_MODEL: dict[type[BaseModel], str] = {
    ConceptRoles: CONCEPTS,
    ValueSetRoles: VALUE_SETS,
    CodeListRoles: CODE_LISTS,
}


def chosen_sections(roles: RolesResult, relations: Collection[str]) -> list[BoundSection]:
    """One section per named relation a binding gives roles to, over its chosen roles."""
    wanted = set(relations)
    found = []
    for entry in roles.structure.bound:
        for target in role_targets(entry.binding):
            if target.relation in wanted and target.relation in roles.refs:
                wanted.discard(target.relation)
                found.append(_section(roles, target, entry.binding.section))
    return found


def _section(roles: RolesResult, target: Target, section: str | None) -> BoundSection:
    bound = chosen_columns(roles.eligibilities, target.relation)
    model = _roles_model(bound, roles.statuses.get(target.relation))
    ref, key, code = roles.refs[target.relation], bound.get(KEY_ROLE), bound.get("code")
    path = _path(target, section)
    if path == CONCEPTS:
        return BoundSection(path, ref, model, group_by=key, identity=key)
    if path == VALUE_SETS:
        return BoundSection(path, ref, model, vocabulary=key, identity=code)
    return BoundSection(path, ref, model, vocabulary=bound.get("vocabulary_id"), identity=code)


def _path(target: Target, section: str | None) -> str:
    paths = {SECTION_OF_MODEL[m] for m in target.models if m in SECTION_OF_MODEL}
    if len(paths) == 1:
        return paths.pop()
    return (section or "").partition(SECTION_SEPARATOR)[0]


def closed_required(roles: RolesResult) -> list[RoleEligibility]:
    """Column roles a decision closed with no column although their binding requires them."""
    required: set[tuple[str, str]] = set()
    for entry in roles.structure.bound:
        for target in role_targets(entry.binding):
            required |= {
                (target.relation, name)
                for model in target.models
                for name, field in model.model_fields.items()
                if field.is_required()
            }
    return [
        e
        for e in roles.eligibilities
        if e.subject is None and e.choice is None and e.source is not None
        and (e.relation, e.role) in required
    ]


def chosen_columns(eligibilities: Iterable[RoleEligibility], relation: str) -> dict[str, str]:
    return {
        e.role: e.column
        for e in eligibilities
        if e.relation == relation and e.subject is None and e.column is not None
    }


def _roles_model(bound: dict[str, str], status: StatusRole | None) -> BaseModel:
    """The chosen roles as a model, a settled status carrying its read value lists."""
    fields: dict[str, Any] = {role: (str, column) for role, column in bound.items()}
    if status is not None and bound.get(lexicon.STATUS_ROLE) == status.column:
        fields[lexicon.STATUS_ROLE] = (StatusRole, status)
    model: type[BaseModel] = create_model(CHOSEN_ROLES, **fields)
    return model()


def with_sources(
    failures: Sequence[Failure],
    eligibilities: Iterable[RoleEligibility],
    checks: Collection[str],
) -> list[Failure]:
    """Each role-check failure on a column names the sources of the entries that chose it."""
    sources: dict[tuple[str, str], set[str]] = {}
    for e in eligibilities:
        if e.subject is None and e.column is not None and e.source is not None:
            sources.setdefault((e.relation, e.column), set()).add(e.source)
    found = []
    for failure in failures:
        if failure.check not in checks or failure.column is None:
            found.append(failure)
            continue
        named = sources.get((failure.relation, failure.column), {SPEC_SOURCE})
        evidence = dict(failure.evidence, source=", ".join(sorted(named)))
        found.append(failure.model_copy(update={"evidence": evidence}))
    return found
