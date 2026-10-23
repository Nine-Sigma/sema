"""Role eligibility: data-fact sets, ordered claims, cross-role exclusion (S5 decisions 1-3, 18).

Per role, its data fact gives the members (every column for a role with none). A human
statement, a stated choice, the run's signature memory, and a unique lexicon match are
tried in that order; an optional role with no candidate left settles ``none`` by code, and
otherwise the role stays open for the role map. No role settles by member count or by the
embedding rank, which is only a fact. A lexicon claim the model contested stays open with
both answers. Roles settle in the lexicon's order and a settled column is excluded from
later roles unless the pair may coincide. Role names come from the spec role models, so
this module spells none of them.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import StatusRole
from sema.intake import lexicon
from sema.intake.eligibility_spec_utils import (
    RolesResult,
    assemble,
    stated_domains,
    stated_roles,
)
from sema.intake.eligibility_target_utils import (
    Target,
    memory_key,
    role_targets,
    side_profile,
)
from sema.intake.eligibility_text_utils import (
    STAGE,
    Dictionary,
    HeaderReading,
    declared_joins,
    is_dictionary_binding,
    note_columns,
    read_dictionary,
    read_header,
    reference_roles,
    resolve_sections,
    vocabularies,
)
from sema.intake.eligibility_utils import (
    Embedder,
    Held,
    Hint,
    RoleEligibility,
    RoleMemory,
    RunDescriptor,
    StatedChoice,
    claimed_columns,
    eligible,
    embedding_rank,
    lexicon_hits,
    scope_of,
    status_classes,
    subject_of,
    with_claims,
    with_stated,
)
from sema.intake.profile import SAMPLE_CAP, ProfileRefusedError, ProfileTools, ValueSample
from sema.intake.shapes import entity_model
from sema.intake.shapes.base import RelationProfile, SlotBinding
from sema.intake.structure import StructureResult
from sema.intake.verify.base import Ambiguity
from sema.log import logger

__all__ = [
    "RolesResult", "Target", "ambiguities", "assemble_spec", "compute", "role_targets", "run_roles",
]

UNRESOLVED_REFERENCE = "unresolved_reference"
WITHHELD_STATUS = "the evidence policy withholds its values until a value ask is approved"


@dataclass(frozen=True)
class Hints:
    """``contested`` are subjects whose lexicon claim the model's role map contested."""

    memory: RoleMemory | None
    embedder: Embedder | None
    choices: Mapping[str, StatedChoice]
    tools: ProfileTools | None = None
    contested: frozenset[str] = field(default_factory=frozenset)


def compute(
    binding: SlotBinding,
    profiles: Iterable[RelationProfile],
    *,
    memory: RoleMemory | None = None,
    embedder: Embedder | None = None,
    stated: Mapping[str, Mapping[str, Any]] | None = None,
    tools: ProfileTools | None = None,
) -> list[RoleEligibility]:
    by_name = {profile.ref.name: profile for profile in profiles}
    hints = Hints(memory, embedder, {}, tools)
    return _compute(binding, by_name, hints, stated or {})


def ambiguities(eligibilities: Iterable[RoleEligibility]) -> list[Ambiguity]:
    """Every open role, every role with more than one member that code did not settle, and
    every open reference. A code settle had no alternative: a slotted column or ``none``
    when no member is free."""
    found = []
    for entry in eligibilities:
        unresolved = [phrase for hint in entry.hints for phrase in hint.unresolved]
        unsettled = entry.choice is None and entry.source is None
        single = len(entry.members) < 2 or entry.source == "code"
        if single and not unresolved and not unsettled:
            continue
        members = entry.members
        if entry.choice is not None and entry.choice not in members:
            members = (*members, entry.choice)
        found.append(Ambiguity(
            stage=STAGE,
            subject=subject_of(entry),
            members=members,
            hints=_hint_texts(entry, unresolved),
            exclusions=dict(entry.exclusions),
            choice=entry.choice,
            confidence=entry.confidence,
            source=entry.source,
        ))
    return found


def run_roles(
    structure: StructureResult,
    profiles: Iterable[RelationProfile],
    tools: ProfileTools,
    *,
    hint: ConceptSourceSpec | None = None,
    embedder: Embedder | None = None,
    choices: Mapping[str, StatedChoice] | None = None,
    contested: Collection[str] = (),
) -> RolesResult:
    """``choices`` are stated choices by ambiguity subject (model answers, decisions); the
    hint's ``domains`` join them, and a choice in ``choices`` wins on the same subject.
    ``contested`` subjects keep no lexicon claim."""
    by_name = {profile.ref.name: profile for profile in profiles}
    stated = stated_roles(hint, structure) if hint is not None else {}
    domains = stated_domains(hint, structure) if hint is not None else {}
    merged = {**domains, **(choices or {})}
    hints = Hints(RoleMemory(), embedder, merged, tools, frozenset(contested))
    keys = hint.entity_model.header_keys if hint is not None and hint.entity_model else None
    models = [e.binding for e in structure.bound if e.binding.template == entity_model.ID]
    found = [role for b in models for role in _compute(b, by_name, hints, stated)]
    headers, dictionaries, notes = _read_dictionaries(models, by_name, found, tools, keys)
    references = [
        with_stated(entry, hints.choices)
        for entry in reference_roles(dictionaries, list(vocabularies(structure, by_name)))
    ]
    joins, open_joins = declared_joins(dictionaries, headers)
    resolved, open_sections = resolve_sections(structure, references, by_name, hints.choices)
    others = [e.binding for e in resolved.bound if e.binding.template != entity_model.ID]
    found += [role for b in others for role in _compute(b, by_name, hints, stated)]
    listed = [*ambiguities([*found, *references]), *open_joins, *open_sections]
    logger.info(
        "roles over {} bindings: {} eligibilities, {} references, {} joins, {} ambiguities",
        len(structure.bound), len(found), len(references), len(joins), len(listed),
    )
    statuses, unverifiable = _statuses(found, by_name, tools, stated)
    return RolesResult(
        eligibilities=(*found, *references),
        ambiguities=tuple(listed),
        joins=tuple(joins),
        headers=headers,
        structure=resolved,
        refs={name: profile.ref for name, profile in by_name.items()},
        statuses=statuses,
        stated=stated,
        notes=notes,
        header_keys=keys,
        unverifiable=unverifiable,
    )


def assemble_spec(descriptor: RunDescriptor, roles: RolesResult) -> ConceptSourceSpec | None:
    return assemble(descriptor, roles)


def _compute(
    binding: SlotBinding,
    by_name: Mapping[str, RelationProfile],
    hints: Hints,
    stated: Mapping[str, Mapping[str, Any]],
) -> list[RoleEligibility]:
    found = []
    for target in role_targets(binding):
        profile = by_name.get(target.relation)
        if profile is not None:
            scope = scope_of(target.relation, target.section)
            side = side_profile(profile, target, hints.tools)
            found += _relation_roles(side, target, hints, stated.get(scope, {}))
    return found


def _relation_roles(
    profile: RelationProfile, target: Target, hints: Hints, stated: Mapping[str, Any]
) -> list[RoleEligibility]:
    key = memory_key(target, profile)
    remembered = hints.memory.recall(key) if hints.memory is not None else None
    plain = _side_roles(profile, target, Hints(None, None, {}), {}, Held({}))
    held = Held(claimed_columns(plain))
    found = _side_roles(profile, target, hints, stated, held, remembered)
    in_force = claimed_columns(found)
    found = [with_claims(entry, in_force) for entry in found]
    if hints.memory is not None and remembered is None:
        hints.memory.remember(key, {e.role: e.choice for e in found if e.choice is not None})
    return found


def _side_roles(
    profile: RelationProfile,
    target: Target,
    hints: Hints,
    stated: Mapping[str, Any],
    held: Held,
    remembered: Mapping[str, str] | None = None,
) -> list[RoleEligibility]:
    fields = {name for model in target.models for name in model.model_fields}
    required = {
        name
        for model in target.models
        for name, field in model.model_fields.items()
        if field.is_required()
    }
    found = []
    for role, column in target.slotted.items():
        slotted = RoleEligibility(
            relation=profile.ref.name, section=target.section, role=role, members=(column,)
        )
        found.append(slotted.settled(column, "code"))
        held.taken[column] = role
    for entry in lexicon.ROLES:
        if entry.name not in fields or entry.name in target.slotted:
            continue
        optional = entry.name not in required
        human = stated.get(entry.name)
        settled = _settle(
            entry, profile, held, remembered, hints, human, optional, target.section
        )
        found.append(settled)
        held.take(settled)
    return found


def _settle(
    entry: lexicon.RoleEntry,
    profile: RelationProfile,
    held: Held,
    remembered: Mapping[str, str] | None,
    hints: Hints,
    stated: Any,
    optional: bool,
    section: str | None = None,
) -> RoleEligibility:
    """An optional role offers ``none`` as its last member, so "no column" is answerable."""
    members = eligible(entry, profile.columns, profile.row_count)
    members = (*members, lexicon.NO_REFERENCE) if optional else members
    excluded = held.excluded(members, entry.name)
    base = RoleEligibility(
        relation=profile.ref.name,
        section=section,
        role=entry.name,
        members=members,
        exclusions=excluded,
        excluded_by=next(iter(excluded.values()), None),
    )
    human = stated.column if isinstance(stated, StatusRole) else stated
    if isinstance(human, str):
        return base.settled(human, "human")
    stated_choice = with_stated(base, hints.choices)
    if stated_choice.source is not None:
        return stated_choice
    recalled = (remembered or {}).get(entry.name)
    if recalled in members:
        return base.hinted(Hint(source="memory", matches=(str(recalled),))).settled(
            str(recalled), "memory"
        )
    remaining = [member for member in members if member not in excluded]
    return _narrow(base, entry, profile, remaining, hints)


def _narrow(
    base: RoleEligibility,
    entry: lexicon.RoleEntry,
    profile: RelationProfile,
    remaining: Sequence[str],
    hints: Hints,
) -> RoleEligibility:
    """A unique lexicon match among the members is a claim unless another role holds it;
    the embedding rank is stated, never settled on."""
    hits = lexicon_hits(entry.name, [m for m in base.members if m != lexicon.NO_REFERENCE])
    if hits:
        base = base.hinted(Hint(source="lexicon", matches=hits))
    if subject_of(base) not in hints.contested:
        if len(hits) == 1 and hits[0] in remaining:
            return base.settled(hits[0], "lexicon")
        if list(remaining) == [lexicon.NO_REFERENCE]:
            return base.settled(lexicon.NO_REFERENCE, "code")
    columns = [column for column in profile.columns if column.name in remaining]
    # A rank orders columns against each other; one column has no rival.
    if hints.embedder is None or len(columns) < 2:
        return base
    return base.hinted(embedding_rank(hints.embedder, entry, columns))


def _read_dictionaries(
    bindings: Sequence[SlotBinding],
    by_name: Mapping[str, RelationProfile],
    found: Sequence[RoleEligibility],
    tools: ProfileTools,
    keys: Any,
) -> tuple[dict[str, HeaderReading], list[Dictionary], dict[str, tuple[str, ...]]]:
    headers: dict[str, HeaderReading] = {}
    dictionaries: list[Dictionary] = []
    notes: dict[str, tuple[str, ...]] = {}
    for binding in filter(is_dictionary_binding, bindings):
        profile = by_name[binding.slots[entity_model.DICTIONARY]]
        header_name = binding.slots.get(entity_model.HEADER)
        header = read_header(tools, by_name[header_name], keys) if header_name else None
        if header is not None:
            headers[header.relation] = header
        choices = {
            e.role: str(e.column)
            for e in found
            if e.relation == profile.ref.name and e.subject is None and e.column is not None
        }
        notes[profile.ref.name] = note_columns(profile, choices.values())
        own_notes = notes[profile.ref.name]
        dictionaries.append(read_dictionary(tools, profile, header, choices, own_notes))
    return headers, dictionaries, notes


def _statuses(
    found: Sequence[RoleEligibility],
    by_name: Mapping[str, RelationProfile],
    tools: ProfileTools,
    stated: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, StatusRole], dict[str, str]]:
    """Value lists for settled status columns, read only when the policy permits, and the
    reason each status the policy withheld stays ``unverifiable``.

    ``SAMPLE_CAP`` (20) exceeds ``STATUS_CARDINALITY_CAP`` (10), so a sample holds every
    value of a column within the cap.
    """
    statuses: dict[str, StatusRole] = {}
    unverifiable: dict[str, str] = {}
    for entry in found:
        column = entry.column
        if entry.role != lexicon.STATUS_ROLE or column is None or entry.source == "human":
            continue
        scope = scope_of(entry.relation, entry.section)
        if lexicon.STATUS_ROLE in stated.get(scope, {}):
            continue
        try:
            sample = tools.sample_values(by_name[entry.relation].ref, column, SAMPLE_CAP)
        except ProfileRefusedError as error:
            logger.info("status values of {} are not read: {}", entry.relation, error)
            unverifiable[scope] = f"its values cannot be read: {error}"
            continue
        if not isinstance(sample, ValueSample):
            unverifiable[scope] = WITHHELD_STATUS
            continue
        classes = status_classes(sample.values)
        if classes is not None:
            active, retired = classes
            statuses[scope] = StatusRole(
                column=column, active_values=active, retired_values=retired
            )
    return statuses, unverifiable


def _hint_texts(entry: RoleEligibility, unresolved: Sequence[str]) -> tuple[str, ...]:
    texts = []
    for hint in entry.hints:
        if hint.source == "embedding" and hint.rank:
            texts.append(f"embedding: {hint.rank[0][0]} margin {hint.margin:.3f}")
        elif hint.matches:
            texts.append(f"{hint.source}: {', '.join(hint.matches)}")
    if unresolved:
        texts.append(UNRESOLVED_REFERENCE)
        texts += [f"{UNRESOLVED_REFERENCE}: {phrase}" for phrase in unresolved]
    texts.append(f"settled_by: {entry.source or 'none'}")
    return tuple(texts)
