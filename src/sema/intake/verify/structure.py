"""Structure verifier: a structure proposal re-proved against the landed relations.

Candidates are recomputed per proposal group over the relations no human pinned and no
ignore removed, so a code-bound binding must be one the predicates produced, survive
precedence, and pass its template's ``verify``; a relation precedence leaves contested
must be on the ambiguity list. Pinned bindings only re-run ``verify``.
"""

from __future__ import annotations

from collections.abc import Sequence

from sema.binding.spec_sections import Provider
from sema.intake.landing import LandResult
from sema.intake.profile import ProfileTools
from sema.intake.shapes import TEMPLATES, apply_precedence, ignore
from sema.intake.shapes.base import RelationProfile, SlotBinding, profile_relations
from sema.intake.shapes.base_utils import failure
from sema.intake.shapes.precedence import claims
from sema.intake.structure import BoundEntry, StructureResult, stage_floors
from sema.intake.structure_utils import group_candidates
from sema.intake.verify.base import Failure, Verdict
from sema.intake.verify.structure_utils import (
    binding_key,
    boilerplate_column,
    partition_counts,
)

PARTITION_COMPLETE = "partition_complete"
UNKNOWN_TEMPLATE = "unknown_template"
IN_ELIGIBILITY_SET = "in_eligibility_set"
TEMPLATE_VERIFIES = "template_verifies"
PRECEDENCE_APPLIED = "precedence_applied"
IGNORE_REASON_KNOWN = "ignore_reason_known"
IGNORE_PROFILE_HOLDS = "ignore_profile_holds"
AMBIGUITY_LIST_COMPLETE = "ambiguity_list_complete"
CHECKS = (
    PARTITION_COMPLETE,
    UNKNOWN_TEMPLATE,
    IN_ELIGIBILITY_SET,
    TEMPLATE_VERIFIES,
    PRECEDENCE_APPLIED,
    IGNORE_REASON_KNOWN,
    IGNORE_PROFILE_HOLDS,
    AMBIGUITY_LIST_COMPLETE,
)
HUMAN = "human"
BOUND_TEMPLATES = {template.ID: template for template in TEMPLATES if template.ID != ignore.ID}


def verify_structure(
    proposal: StructureResult,
    land: LandResult,
    tools: ProfileTools,
    *,
    provider: Provider = "staging",
) -> Verdict:
    profiles = profile_relations(land, tools, provider)
    by_name = {profile.ref.name: profile for profile in profiles}
    failures = _partition_failures(proposal, by_name)
    failures += _template_failures(proposal)
    failures += _ignore_failures(proposal.ignored, profiles, tools)
    known = [e for e in proposal.bound if e.binding.template in BOUND_TEMPLATES]
    pinned = [e for e in known if e.source == HUMAN]
    for entry in pinned:
        failures += _verified(entry.binding, profiles, tools)
    removed = {name for e in proposal.bound if e.source == HUMAN for name in e.binding.relations}
    removed |= {name for binding in proposal.ignored for name in binding.relations}
    listed = {ambiguity.subject for ambiguity in proposal.ambiguities}
    coded = [e for e in known if e.source != HUMAN]
    for group in proposal.groups:
        members = [by_name[name] for name in group if name in by_name and name not in removed]
        failures += _group_failures(members, coded, known, listed, tools)
    return Verdict.of(
        failures=failures, ambiguities=proposal.ambiguities, floors=stage_floors()
    )


def _partition_failures(
    proposal: StructureResult, by_name: dict[str, RelationProfile]
) -> list[Failure]:
    counts = partition_counts(proposal)
    names = list(dict.fromkeys([*by_name, *counts]))
    return [
        failure(PARTITION_COMPLETE, name, appearances=counts[name], landed=name in by_name)
        for name in names
        if counts[name] != 1 or name not in by_name
    ]


def _template_failures(proposal: StructureResult) -> list[Failure]:
    stated = [(e.binding, e.binding.template in BOUND_TEMPLATES) for e in proposal.bound]
    stated += [(binding, binding.template == ignore.ID) for binding in proposal.ignored]
    return [
        failure(UNKNOWN_TEMPLATE, name, template=binding.template)
        for binding, known in stated
        if not known
        for name in dict.fromkeys(binding.relations)
    ]


def _ignore_failures(
    ignored: Sequence[SlotBinding], profiles: Sequence[RelationProfile], tools: ProfileTools
) -> list[Failure]:
    by_name = {profile.ref.name: profile for profile in profiles}
    failures = []
    for binding in ignored:
        relation = binding.relations[0] if binding.relations else ""
        claimed = binding.slots.get(ignore.REASON)
        if claimed not in ignore.REASONS:
            failures.append(failure(IGNORE_REASON_KNOWN, relation, reason=claimed or ""))
            continue
        profile = by_name.get(relation)
        if profile is None or binding.template != ignore.ID:
            continue
        found = ignore.reason_for(profile, profiles, tools)
        if found == claimed:
            continue
        boilerplate = claimed == ignore.BOILERPLATE
        column = boilerplate_column(profile, profiles, tools) if boilerplate else None
        failures.append(
            failure(IGNORE_PROFILE_HOLDS, relation, column, claimed=claimed, found=found or "")
        )
    return failures


def _group_failures(
    members: Sequence[RelationProfile],
    coded: Sequence[BoundEntry],
    known: Sequence[BoundEntry],
    listed: set[str],
    tools: ProfileTools,
) -> list[Failure]:
    names = {profile.ref.name for profile in members}
    found = group_candidates(members, tools)
    keys = {binding_key(binding) for binding in found}
    claimed = {
        name
        for binding in [*found, *(entry.binding for entry in known)]
        if claims(binding)
        for name in binding.relations
    }
    failures = []
    for entry in (e for e in coded if set(e.binding.relations) & names):
        binding = entry.binding
        if binding_key(binding) not in keys:
            relations = dict.fromkeys(binding.relations)
            failures += [
                failure(IN_ELIGIBILITY_SET, name, template=binding.template) for name in relations
            ]
        if not claims(binding):
            failures += [
                failure(PRECEDENCE_APPLIED, name, template=binding.template)
                for name in dict.fromkeys(binding.relations)
                if name in claimed
            ]
        failures += _verified(binding, members, tools)
    contested = [a.subject for a in apply_precedence(found).ambiguities]
    failures += [
        failure(AMBIGUITY_LIST_COMPLETE, name, members=len(_over(found, name)))
        for name in contested
        if name not in listed
    ]
    return failures


def _verified(
    binding: SlotBinding, group: Sequence[RelationProfile], tools: ProfileTools
) -> list[Failure]:
    verdict = BOUND_TEMPLATES[binding.template].verify(binding, list(group), tools)
    return [
        failure(
            TEMPLATE_VERIFIES,
            found.relation or (binding.relations[0] if binding.relations else ""),
            found.column,
            **{**found.evidence, "template": binding.template, "template_check": found.check},
        )
        for found in verdict.failures
    ]


def _over(bindings: Sequence[SlotBinding], name: str) -> list[SlotBinding]:
    return [binding for binding in bindings if name in binding.relations]
