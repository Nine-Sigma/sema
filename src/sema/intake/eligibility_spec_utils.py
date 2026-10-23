"""Spec assembly from settled roles and a run descriptor (design Decision 20).

Fixed rules supply what the data cannot: a code list's ``vocabulary_id`` is its section
name, which Roles set from the relation's file stem; a per-file entity is named by its
header (the importer reads it), so entries carry no ``entity``; ``notes`` is every text
column of the dictionary no other role took; declared joins and resolved domains become
each entry's ``foreign_keys`` and ``domains``. A concept section groups by its key.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import (
    CodeListRoles,
    CodeListSpec,
    ConceptRoles,
    ConceptsSpec,
    DeclaredForeignKey,
    EntityModelRoles,
    EntityModelSpec,
    HeaderKeys,
    LevelColumnsHierarchy,
    PerFileEntry,
    RelationRef,
    SchemeSpec,
    StatusRole,
    TermStorageSpec,
    ValueSetRoles,
    ValueSetsSpec,
)
from sema.intake import lexicon
from sema.intake.eligibility_text_utils import (
    CODE_LISTS,
    CONCEPTS,
    VOCABULARY_ROLE,
    HeaderReading,
)
from sema.intake.eligibility_utils import (
    HUMAN,
    SOURCE_CONFIDENCE,
    DeclaredJoin,
    RoleEligibility,
    RunDescriptor,
    StatedChoice,
    scope_of,
    subject_of,
)
from sema.intake.loaders import athena, semsql_utils
from sema.intake.shapes import (
    athena_bundle,
    concept_and_value_set,
    concept_list,
    concept_with_values_denormalized,
    entity_model,
    filled_sections,
    level_columns,
    semsql,
)
from sema.intake.shapes.base import SlotBinding
from sema.intake.shapes.base_utils import section_of
from sema.intake.structure import StructureResult
from sema.intake.structure_hint_utils import KEY_ROLE, SECTION_SEPARATOR, section_at
from sema.intake.verify.base import Ambiguity
from sema.log import logger

ASSEMBLED_SPEC_VERSION = 2
VALUE_SETS = section_of(ValueSetsSpec)
HIERARCHY = section_of(LevelColumnsHierarchy)
# Sections whose bindings add up rather than compete; every other section takes one binding.
MERGED_SECTIONS = frozenset({CODE_LISTS})
# Templates whose bindings assembly builds into one section together.
JOINT_TEMPLATES = frozenset({entity_model.ID})
Roles = TypeVar("Roles", bound=BaseModel)
Part = dict[str, Any]
LoaderSections = Callable[[Mapping[str, RelationRef], str], dict[str, BaseModel]]


@dataclass(frozen=True)
class RolesResult:
    eligibilities: tuple[RoleEligibility, ...]
    ambiguities: tuple[Ambiguity, ...]
    joins: tuple[DeclaredJoin, ...]
    headers: dict[str, HeaderReading]
    structure: StructureResult
    refs: dict[str, RelationRef]
    statuses: dict[str, StatusRole] = field(default_factory=dict)
    stated: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    header_keys: HeaderKeys | None = None
    unverifiable: dict[str, str] = field(default_factory=dict)
    disagreements: dict[str, int] = field(default_factory=dict)
    contested: dict[str, str] = field(default_factory=dict)


def assemble(descriptor: RunDescriptor, roles: RolesResult) -> ConceptSourceSpec | None:
    structure = roles.structure
    if structure.ambiguities or any(not filled_sections(e.binding) for e in structure.bound):
        logger.info("no spec assembled: the structure still lists open bindings or sections")
        return None
    parts: list[Part | None] = []
    models = [e.binding for e in structure.bound if e.binding.template == entity_model.ID]
    if models:
        parts.append(_entity_model(models, roles, descriptor))
    for entry in structure.bound:
        builder = BUILDERS.get(entry.binding.template)
        if entry.binding.template in LOADER_SECTIONS:
            parts.append(_loader_fixed(entry.binding, roles, descriptor.identity_system))
        elif builder is not None:
            parts.append(builder(entry.binding, roles))
        elif entry.binding.template != entity_model.ID:
            logger.info("no spec assembled: {} has no assembly rule", entry.binding.template)
            return None
    sections = _merge(parts)
    if sections is None:
        return None
    try:
        return ConceptSourceSpec(
            spec_version=ASSEMBLED_SPEC_VERSION,
            id=descriptor.id,
            identity_system=descriptor.identity_system,
            release=descriptor.release,
            default_provider=next(iter(roles.refs.values())).provider,
            shape=structure.to_shape(),
            scheme=_scheme(descriptor) if CONCEPTS in sections else descriptor.scheme,
            term_storage=descriptor.term_storage or TermStorageSpec(),
            **sections,
        )
    except ValidationError as error:
        logger.info("no spec assembled: the sections do not validate: {}", error)
        return None


def _scheme(descriptor: RunDescriptor) -> SchemeSpec:
    if descriptor.scheme is not None:
        return descriptor.scheme
    return SchemeSpec(
        qualified_name=f"{descriptor.identity_system}.{descriptor.id}",
        description=(
            f"Concept source {descriptor.id}, release {descriptor.release}, "
            "assembled without a stated scheme"
        ),
    )


def stated_roles(hint: ConceptSourceSpec, structure: StructureResult) -> dict[str, dict[str, Any]]:
    """Roles the hint states, by relation; each becomes a ``source: human`` choice."""
    stated: dict[str, dict[str, Any]] = {}
    for entry in structure.bound:
        binding = entry.binding
        section = section_at(hint, binding.section) if binding.section else None
        for relation, model in _stated_models(binding, section, hint):
            stated[relation] = {
                name: getattr(model, name)
                for name in model.model_fields_set
                if getattr(model, name) is not None
            }
    return stated


def stated_domains(hint: ConceptSourceSpec, structure: StructureResult) -> dict[str, StatedChoice]:
    """The hint's per-file ``domains`` as human reference choices, keyed by ambiguity subject."""
    dictionaries = {
        e.binding.slots.get(entity_model.DICTIONARY)
        for e in structure.bound
        if e.binding.template == entity_model.ID
    }
    entries = hint.entity_model.entities if hint.entity_model is not None else None
    return {
        subject_of(RoleEligibility(
            relation=entry.relation.name, role=VOCABULARY_ROLE, subject=prop, members=()
        )): StatedChoice(choice=vocabulary, source=HUMAN, confidence=SOURCE_CONFIDENCE[HUMAN])
        for entry in entries or ()
        if entry.relation.name in dictionaries
        for prop, vocabulary in entry.domains.items()
    }


def _stated_models(
    binding: SlotBinding, section: object, hint: ConceptSourceSpec
) -> list[tuple[str, BaseModel]]:
    roles = getattr(section, "roles", None)
    if not isinstance(roles, BaseModel):
        return []
    if binding.template == concept_with_values_denormalized.ID:
        relation = binding.relations[0]
        values = hint.value_sets
        valued = [(scope_of(relation, VALUE_SETS), values.roles)] if values is not None else []
        return [(relation, roles), *valued]
    if binding.template != concept_and_value_set.ID:
        return [(binding.relations[0], roles)]
    found: list[tuple[str, BaseModel]] = [
        (binding.slots[concept_and_value_set.CONCEPT_RELATION], roles)
    ]
    if hint.value_sets is not None:
        found.append((binding.slots[concept_and_value_set.VALUE_RELATION], hint.value_sets.roles))
    return found


def _merge(parts: Sequence[Part | None]) -> Part | None:
    merged: Part = {}
    for part in parts:
        if part is None:
            return None
        for name, value in part.items():
            if name in MERGED_SECTIONS:
                merged[name] = [*merged.get(name, []), *value]
            elif name in merged:
                logger.info("no spec assembled: two bindings fill section {}", name)
                return None
            else:
                merged[name] = value
    return merged


def roles_of(
    relation: str,
    model: type[Roles],
    roles: RolesResult,
    extra: Mapping[str, Any] | None = None,
    section: str | None = None,
) -> Roles | None:
    """``section`` picks one side of a relation that feeds two sections. A status the
    policy kept unread stays bound with empty value lists, marked ``unverifiable``."""
    scope = scope_of(relation, section)
    values: dict[str, Any] = {
        e.role: e.column
        for e in roles.eligibilities
        if e.relation == relation and e.section == section and e.subject is None
        and e.column is not None and e.role in model.model_fields
    }
    if lexicon.STATUS_ROLE in values:
        column = values.pop(lexicon.STATUS_ROLE)
        status = roles.statuses.get(scope)
        if status is None and scope in roles.unverifiable:
            status = StatusRole(column=column, active_values=[], retired_values=[])
        values |= {lexicon.STATUS_ROLE: status} if status is not None else {}
    values |= dict(extra or {})
    values |= {k: v for k, v in roles.stated.get(scope, {}).items() if k in model.model_fields}
    missing = [n for n, f in model.model_fields.items() if f.is_required() and n not in values]
    if missing:
        logger.info("no spec assembled: {} has no settled {}", relation, missing)
        return None
    return model.model_validate(values)


def _loader_fixed(binding: SlotBinding, roles: RolesResult, identity_system: str) -> Part | None:
    relations = {slot: roles.refs[name] for slot, name in binding.slots.items()}
    rendered = LOADER_SECTIONS[binding.template](relations, identity_system)
    sections = filled_sections(binding)
    missing = [name for name in sections if name not in rendered]
    if missing:
        logger.info("no spec assembled: {} renders no {}", binding.template, missing)
        return None
    return {name: rendered[name] for name in sections}


def _concept_list(binding: SlotBinding, roles: RolesResult) -> Part | None:
    relation = binding.slots[concept_list.RELATION]
    head, _, vocabulary = (binding.section or "").partition(SECTION_SEPARATOR)
    ref = roles.refs[relation]
    if head == CODE_LISTS:
        listed = roles_of(relation, CodeListRoles, roles)
        if listed is None:
            return None
        return {CODE_LISTS: [CodeListSpec(vocabulary_id=vocabulary, relation=ref, roles=listed)]}
    concept = roles_of(relation, ConceptRoles, roles)
    if head != CONCEPTS or concept is None:
        return None
    group_by = getattr(concept, KEY_ROLE)
    return {CONCEPTS: ConceptsSpec(relation=ref, group_by=group_by, roles=concept)}


def _concept_and_value_set(binding: SlotBinding, roles: RolesResult) -> Part | None:
    template = concept_and_value_set
    concepts = binding.slots[template.CONCEPT_RELATION]
    values = binding.slots[template.VALUE_RELATION]
    concept_roles = roles_of(concepts, ConceptRoles, roles)
    value_roles = roles_of(values, ValueSetRoles, roles)
    if concept_roles is None or value_roles is None:
        return None
    group_by = binding.slots[template.CONCEPT_KEY]
    return {
        CONCEPTS: ConceptsSpec(
            relation=roles.refs[concepts], group_by=group_by, roles=concept_roles
        ),
        VALUE_SETS: ValueSetsSpec(relation=roles.refs[values], roles=value_roles),
    }


def _denormalized(binding: SlotBinding, roles: RolesResult) -> Part | None:
    template = concept_with_values_denormalized
    relation = binding.slots[template.RELATION]
    concept_roles = roles_of(relation, ConceptRoles, roles)
    value_roles = roles_of(relation, ValueSetRoles, roles, section=VALUE_SETS)
    if concept_roles is None or value_roles is None:
        return None
    ref = roles.refs[relation]
    group_by = binding.slots[template.GROUP_BY]
    return {
        CONCEPTS: ConceptsSpec(relation=ref, group_by=group_by, roles=concept_roles),
        VALUE_SETS: ValueSetsSpec(relation=ref, roles=value_roles),
    }


def _level_columns(binding: SlotBinding, roles: RolesResult) -> Part | None:
    prefix = level_columns.LEVEL
    levels = sorted(
        (int(slot.removeprefix(prefix)), column)
        for slot, column in binding.slots.items()
        if slot.startswith(prefix)
    )
    relation = roles.refs[binding.slots[level_columns.RELATION]]
    columns = [column for _, column in levels]
    return {HIERARCHY: LevelColumnsHierarchy(level_columns=columns, relation=relation)}


def _entity_model(
    bindings: Sequence[SlotBinding], roles: RolesResult, descriptor: RunDescriptor
) -> Part | None:
    template = entity_model
    case = descriptor.entity_case or EntityModelSpec.model_fields["entity_case"].default
    if any(b.slots.get(template.LAYOUT) != template.PER_FILE for b in bindings):
        if len(bindings) != 1:
            return None
        relation = bindings[0].slots[template.RELATION]
        single = roles_of(relation, EntityModelRoles, roles)
        if single is None:
            return None
        spec = EntityModelSpec(relation=roles.refs[relation], roles=single, entity_case=case)
        return {template.SECTION: spec}
    found = [
        roles_of(d, EntityModelRoles, roles, {lexicon.NOTES_ROLE: list(roles.notes.get(d, ()))})
        for d in (b.slots[template.DICTIONARY] for b in bindings)
    ]
    shared = [model for model in found if model is not None]
    if len(shared) != len(found) or len({model.model_dump_json() for model in shared}) != 1:
        logger.info("no spec assembled: the dictionaries do not share one role map")
        return None
    spec = EntityModelSpec.model_validate(dict(
        layout=template.PER_FILE,
        entities=[_entry(b, roles) for b in bindings],
        header_keys=roles.header_keys,
        roles=shared[0],
        entity_case=case,
    ))
    return {template.SECTION: spec}


def _entry(binding: SlotBinding, roles: RolesResult) -> PerFileEntry:
    dictionary = binding.slots[entity_model.DICTIONARY]
    header = binding.slots.get(entity_model.HEADER)
    keys = [
        DeclaredForeignKey(
            columns=list(join.columns), entity=join.to_entity, reference_columns=list(join.columns)
        )
        for join in roles.joins
        if join.from_relation == dictionary
    ]
    domains = {
        str(e.subject): str(e.choice)
        for e in roles.eligibilities
        if e.relation == dictionary and e.role == VOCABULARY_ROLE and e.subject is not None
        and e.choice not in (None, lexicon.NO_REFERENCE)
    }
    return PerFileEntry(
        relation=roles.refs[dictionary],
        header=roles.refs[header] if header else None,
        foreign_keys=keys,
        domains=domains,
    )


BUILDERS: Mapping[str, Callable[[SlotBinding, RolesResult], Part | None]] = {
    concept_list.ID: _concept_list,
    concept_and_value_set.ID: _concept_and_value_set,
    concept_with_values_denormalized.ID: _denormalized,
    level_columns.ID: _level_columns,
}
# The loader writes these relations itself, so it renders their sections; Roles binds none.
LOADER_SECTIONS: Mapping[str, LoaderSections] = {
    semsql.ID: semsql_utils.fixed_sections,
    athena_bundle.ID: athena.fixed_sections,
}
