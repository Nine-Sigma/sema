"""Roles verifier: the spec's role bindings re-proved against eligibility and the landed data.

Binding checks compare each bound column with the role's predicate set from the Roles
stage; value checks run each role's predicate over the bound relation as counts. Without
a spec only the stage's own sections and ambiguity list are checked. A required role a
decision closed with no column fails ``required_roles_bound`` naming that decision's source.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from itertools import combinations

from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import ExternalRefRole, MappingsSpec, RelationRef
from sema.intake import lexicon
from sema.intake.eligibility import ambiguities
from sema.intake.eligibility_spec_utils import RolesResult
from sema.intake.eligibility_text_utils import STAGE, VOCABULARY_ROLE
from sema.intake.eligibility_utils import RoleEligibility
from sema.intake.profile import SAMPLE_CAP, ProfileTools, ValueSample
from sema.intake.shapes import filled_sections
from sema.intake.shapes.base import JOIN_RESOLVE_FLOOR
from sema.intake.shapes.base_utils import JOIN_FLOOR, failure, section_of
from sema.intake.structure import StructureResult
from sema.intake.structure_hint_utils import section_at
from sema.intake.verify.base import Ambiguity, Failure, FloorValue, Verdict
from sema.intake.verify.roles_chosen_utils import (
    chosen_sections,
    closed_required,
    with_sources,
)
from sema.intake.verify.roles_utils import (
    GROUP_KEY_ROLES,
    VALUE_CHECKS,
    BoundSection,
    bound_sections,
    constant_group_key,
    role_columns,
    value_failures,
)

REQUIRED_ROLES_BOUND = "required_roles_bound"
COLUMN_EXISTS = "column_exists"
IN_ELIGIBILITY_SET = "in_eligibility_set"
ONE_ROLE_PER_COLUMN = "one_role_per_column"
SECTION_RESOLVED = "section_resolved"
AMBIGUITY_LIST_COMPLETE = "ambiguity_list_complete"
EXTERNAL_REFS_RESOLVE = "external_refs_resolve"
CHECKS = (
    REQUIRED_ROLES_BOUND,
    COLUMN_EXISTS,
    IN_ELIGIBILITY_SET,
    ONE_ROLE_PER_COLUMN,
    SECTION_RESOLVED,
    AMBIGUITY_LIST_COMPLETE,
    *VALUE_CHECKS,
    EXTERNAL_REFS_RESOLVE,
)
FLOORS: dict[str, FloorValue] = {
    "DISPLAY_NULL_RATE_FLOOR": lexicon.DISPLAY_NULL_RATE_FLOOR,
    **JOIN_FLOOR,
}
ROLE_CHECKS = (COLUMN_EXISTS, IN_ELIGIBILITY_SET, ONE_ROLE_PER_COLUMN, *VALUE_CHECKS)
UNVERIFIABLE = "unverifiable"
MAPPINGS = section_of(MappingsSpec)


def verify_roles(
    roles: RolesResult,
    tools: ProfileTools,
    *,
    spec: ConceptSourceSpec | None,
    relations: Collection[str] = (),
) -> Verdict:
    """``relations`` also re-proves each named relation's chosen roles, with no spec."""
    failures = _section_failures(roles.structure)
    failures += _ambiguity_failures(roles)
    failures += [
        failure(REQUIRED_ROLES_BOUND, e.relation, role=e.role, source=e.source or "")
        for e in closed_required(roles)
    ]
    listed: list[Ambiguity] = list(roles.ambiguities)
    sections = chosen_sections(roles, relations) if relations else []
    if spec is not None:
        failures += _required_failures(roles.structure, spec)
        sections += bound_sections(spec)
        found, unverifiable = _external_refs(spec, tools)
        failures += found
        listed += unverifiable
    for section in sections:
        failures += _section_role_failures(section, roles.eligibilities, tools)
    failures = with_sources(failures, roles.eligibilities, ROLE_CHECKS)
    if spec is not None:
        failures += _domain_failures(spec, roles.eligibilities)
    return Verdict.of(failures=failures, ambiguities=listed, floors=FLOORS)


def _section_role_failures(
    section: BoundSection, eligibilities: Sequence[RoleEligibility], tools: ProfileTools
) -> list[Failure]:
    profile = tools.describe_relation(section.relation)
    present = {column.name for column in profile.columns}
    failures = _binding_failures(section, present, eligibilities, tools)
    return failures + value_failures(section, profile, tools)


def _section_failures(structure: StructureResult) -> list[Failure]:
    return [
        failure(SECTION_RESOLVED, entry.binding.relations[0], template=entry.binding.template)
        for entry in structure.bound
        if not filled_sections(entry.binding) and entry.binding.relations
    ]


def _ambiguity_failures(roles: RolesResult) -> list[Failure]:
    listed = {ambiguity.subject for ambiguity in roles.ambiguities}
    return [
        failure(AMBIGUITY_LIST_COMPLETE, entry.relation, role=entry.role, subject=found.subject)
        for entry in roles.eligibilities
        for found in ambiguities([entry])
        if found.subject not in listed
    ]


def _required_failures(structure: StructureResult, spec: ConceptSourceSpec) -> list[Failure]:
    return [
        failure(
            REQUIRED_ROLES_BOUND,
            entry.binding.relations[0],
            section=entry.binding.section,
            template=entry.binding.template,
        )
        for entry in structure.bound
        if entry.binding.section is not None
        and entry.binding.relations
        and section_at(spec, entry.binding.section) is None
    ]


def _binding_failures(
    section: BoundSection,
    present: set[str],
    eligibilities: Sequence[RoleEligibility],
    tools: ProfileTools,
) -> list[Failure]:
    relation = section.relation.name
    bound = role_columns(section.roles)
    sets = _eligibility_sets(eligibilities, relation)
    failures = [
        failure(COLUMN_EXISTS, relation, column, role=role)
        for role, column in bound.items()
        if column not in present
    ]
    failures += [
        failure(IN_ELIGIBILITY_SET, relation, column, role=role, members=sorted(sets[role]))
        for role, column in bound.items()
        if column in present and role in sets and column not in sets[role]
    ]
    by_column: dict[str, list[str]] = {}
    for role, column in bound.items():
        by_column.setdefault(column, []).append(role)
    failures += [
        failure(ONE_ROLE_PER_COLUMN, relation, column, roles=held)
        for column, held in by_column.items()
        if not _may_share(held) and not _shared_group_key(section, column, held, tools)
    ]
    return failures


def _eligibility_sets(
    eligibilities: Iterable[RoleEligibility], relation: str
) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    for entry in eligibilities:
        if entry.relation == relation and entry.subject is None:
            sets.setdefault(entry.role, set()).update(entry.members)
    return sets


def _domain_failures(
    spec: ConceptSourceSpec, eligibilities: Iterable[RoleEligibility]
) -> list[Failure]:
    """Each bound domain re-proved against the reference entry for its relation and property."""
    references = {
        (e.relation, e.subject): e
        for e in eligibilities
        if e.role == VOCABULARY_ROLE and e.subject is not None
    }
    entities = spec.entity_model.entities if spec.entity_model is not None else None
    return [
        failure(
            IN_ELIGIBILITY_SET, entry.relation.name, prop, role=VOCABULARY_ROLE,
            vocabulary=vocabulary, members=sorted(reference.members), source=reference.source or "",
        )
        for entry in entities or ()
        for prop, vocabulary in entry.domains.items()
        if (reference := references.get((entry.relation.name, prop))) is not None
        and vocabulary not in reference.members
    ]


def _may_share(held: Sequence[str]) -> bool:
    return all(lexicon.coincide(first, second) for first, second in combinations(held, 2))


def _shared_group_key(
    section: BoundSection, column: str, held: Sequence[str], tools: ProfileTools
) -> bool:
    return frozenset(held) == GROUP_KEY_ROLES and constant_group_key(section, column, tools)


def _external_refs(
    spec: ConceptSourceSpec, tools: ProfileTools
) -> tuple[list[Failure], list[Ambiguity]]:
    """Reference systems resolve above the floor when loaded and readable, else unverifiable."""
    failures: list[Failure] = []
    open_refs: list[Ambiguity] = []
    for details in (d for d in (spec.concept_details, spec.term_details) if d is not None):
        for ref in details.roles.external_refs:
            subject = f"{details.relation.name}.{ref.column}"
            rate = _resolve_rate(details.relation, ref, tools)
            if rate is None:
                open_refs.append(_unverifiable(subject, ref.system))
            elif rate < JOIN_RESOLVE_FLOOR:
                failures.append(failure(
                    EXTERNAL_REFS_RESOLVE, details.relation.name, ref.column,
                    system=ref.system, resolve_rate=rate,
                ))
    if spec.mappings is not None:
        subject = f"{spec.mappings.relation.name}.{MAPPINGS}"
        open_refs.append(_unverifiable(subject, spec.mappings.roles.object_id))
    return failures, open_refs


def _resolve_rate(relation: RelationRef, ref: ExternalRefRole, tools: ProfileTools) -> float | None:
    if ref.system not in tools.loaded_systems or tools.vocabulary is None:
        return None
    sample = tools.sample_values(relation, ref.column, SAMPLE_CAP)
    if not isinstance(sample, ValueSample):
        return None
    codes = [str(value) for value in sample.values]
    return tools.resolve_external_refs(ref.system, codes).resolve_rate


def _unverifiable(subject: str, member: str) -> Ambiguity:
    return Ambiguity(stage=STAGE, subject=subject, members=(member,), hints=(UNVERIFIABLE,))
