"""Structure verifier: a structure proposal re-proved against the landed relations.

Candidates are recomputed per proposal group over the relations no human pinned and no
ignore removed, so a code-bound binding must be one the predicates produced, survive
precedence, and pass its template's ``verify``; a relation precedence leaves contested
must be on the ambiguity list. Only claims in force (kept by precedence, or bound) take a
relation from a single-relation binding. Pinned bindings, and model proposals over a
proposal target or a view, only re-run ``verify`` over what they were stated over.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from sema.binding.spec_sections import Provider
from sema.intake.structure.described_data_utils import (
    described_by,
    per_file_dictionaries,
    uses_described,
)
from sema.intake.land.landing import LandResult
from sema.intake.land.profile import ProfileTools
from sema.intake.structure.shapes import TEMPLATES, apply_precedence, ignore
from sema.intake.structure.shapes.base import RelationProfile, SlotBinding, profile_relations
from sema.intake.structure.shapes.base_utils import failure
from sema.intake.structure.shapes.precedence import claims
from sema.intake.structure.structure import (
    BoundEntry,
    StructureResult,
    stage_floor_classes,
    stage_floors,
)
from sema.intake.structure.structure_utils import group_candidates
from sema.intake.verify.base import Failure, Verdict
from sema.intake.verify.structure_utils import (
    binding_key,
    boilerplate_column,
    partition_counts,
)
from sema.intake.structure.views.verify_utils import verify_over

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
MODEL = "model"
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
    targets = set(proposal.proposal_targets)
    stated = [e for e in known if e.source == HUMAN or _proposed(e, targets)]
    for entry in stated:
        failures += _verified(entry, profiles, tools)
    described = _described(proposal, by_name, tools)
    failures += _described_failures(stated, described)
    failures += _shared_claim_failures(known)
    removed = {name for e in proposal.bound if e.source == HUMAN for name in e.binding.relations}
    removed |= {name for e in stated for name in e.binding.relations}
    removed |= {name for binding in proposal.ignored for name in binding.relations}
    listed = {a.subject for a in (*proposal.ambiguities, *proposal.settled)}
    coded = [e for e in known if e.source != HUMAN and e not in stated]
    for group in proposal.groups:
        members = [by_name[name] for name in group if name in by_name and name not in removed]
        failures += _group_failures(members, coded, known, listed, tools, described)
    return Verdict.of(
        failures=failures,
        ambiguities=proposal.ambiguities,
        floors=stage_floors(),
        floor_classes=stage_floor_classes(),
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
        if profile is None or binding.template != ignore.ID or claimed == ignore.HUMAN:
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
    described: dict[str, str],
) -> list[Failure]:
    names = {profile.ref.name for profile in members}
    found = [b for b in group_candidates(members, tools) if not uses_described(b, described)]
    decided = apply_precedence(found)
    keys = {binding_key(binding) for binding in found}
    in_force = [*decided.candidates, *(entry.binding for entry in known)]
    claimed = {name for binding in in_force if claims(binding) for name in binding.relations}
    failures = []
    for entry in (e for e in coded if set(e.binding.relations) & names):
        binding = entry.binding
        if binding_key(binding) not in keys:
            relations = dict.fromkeys(binding.relations)
            failures += [
                failure(IN_ELIGIBILITY_SET, name, template=binding.template, source=entry.source)
                for name in relations
            ]
        if not claims(binding):
            failures += [
                failure(PRECEDENCE_APPLIED, name, template=binding.template, source=entry.source)
                for name in dict.fromkeys(binding.relations)
                if name in claimed
            ]
        failures += _verified(entry, members, tools)
    contested = [a.subject for a in decided.ambiguities]
    failures += [
        failure(AMBIGUITY_LIST_COMPLETE, name, members=len(_over(found, name)))
        for name in contested
        if name not in listed
    ]
    return failures


def _shared_claim_failures(known: Sequence[BoundEntry]) -> list[Failure]:
    """Two bound claims over one relation: each claiming entry fails, naming its source."""
    claiming = [entry for entry in known if claims(entry.binding)]
    counts = Counter(name for e in claiming for name in dict.fromkeys(e.binding.relations))
    return [
        failure(
            PRECEDENCE_APPLIED,
            name,
            template=entry.binding.template,
            source=entry.source,
            claims=counts[name],
        )
        for entry in claiming
        for name in dict.fromkeys(entry.binding.relations)
        if counts[name] > 1
    ]


def _described(
    proposal: StructureResult, by_name: dict[str, RelationProfile], tools: ProfileTools
) -> dict[str, str]:
    """Relations a bound dictionary describes, over the whole namespace."""
    names = per_file_dictionaries(entry.binding for entry in proposal.bound)
    dictionaries = [by_name[name] for name in names if name in by_name]
    return described_by(dictionaries, list(by_name.values()), tools)


def _described_failures(
    stated: Sequence[BoundEntry], described: dict[str, str]
) -> list[Failure]:
    """A pinned or proposed binding that reads described data as a dictionary or header."""
    return [
        failure(
            IN_ELIGIBILITY_SET,
            name,
            template=entry.binding.template,
            source=entry.source,
            described_by=described[name],
        )
        for entry in stated
        if uses_described(entry.binding, described)
        for name in dict.fromkeys(entry.binding.relations)
        if name in described
    ]


def _proposed(entry: BoundEntry, targets: set[str]) -> bool:
    """A model proposal: over a view, or over a relation no code candidate could bind."""
    spans_target = bool(targets & set(entry.binding.relations))
    return entry.source == MODEL and (entry.view is not None or spans_target)


def _verified(
    entry: BoundEntry, group: Sequence[RelationProfile], tools: ProfileTools
) -> list[Failure]:
    binding = entry.binding
    template = BOUND_TEMPLATES[binding.template]
    return [
        failure(
            TEMPLATE_VERIFIES,
            found.relation or (binding.relations[0] if binding.relations else ""),
            found.column,
            **found.evidence,
            **dict(template=binding.template, template_check=found.check, source=entry.source),
        )
        for found in verify_over(template, binding, entry.view, group, tools)
    ]


def _over(bindings: Sequence[SlotBinding], name: str) -> list[SlotBinding]:
    return [binding for binding in bindings if name in binding.relations]
