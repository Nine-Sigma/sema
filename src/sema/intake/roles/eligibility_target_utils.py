"""Role targets: the role models a bound template gives each of its relations.

A denormalized binding feeds two sections from one relation: the group's own attributes
(columns ``group_by`` determines) take concept roles and the rows take value-set roles.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel

from sema.binding.spec_sections import (
    CodeListRoles,
    ConceptRoles,
    EntityModelRoles,
    ValueSetRoles,
    ValueSetsSpec,
)
from sema.intake.roles import lexicon
from sema.intake.roles.eligibility_text_utils import CODE_LISTS, CONCEPTS, is_dictionary_binding
from sema.intake.roles.eligibility_utils import MemoryKey, scope_of
from sema.intake.land.profile import ProfileTools
from sema.intake.structure.shapes import (
    concept_and_value_set,
    concept_list,
    concept_with_values_denormalized,
    entity_model,
)
from sema.intake.structure.shapes.base import RelationProfile, SlotBinding
from sema.intake.structure.shapes.base_utils import section_of
from sema.intake.structure.structure_hint_utils import KEY_ROLE, SECTION_SEPARATOR

Split = Literal["group", "rows"]
VALUE_SETS = section_of(ValueSetsSpec)


@dataclass(frozen=True)
class Target:
    """``section`` qualifies the subject when the relation feeds another section too;
    ``split`` keeps the columns on one side of the key's functional dependencies."""

    template: str
    relation: str
    models: tuple[type[BaseModel], ...]
    slotted: dict[str, str]
    section: str | None = None
    split: Split | None = None


def role_targets(binding: SlotBinding) -> list[Target]:
    slots, template = binding.slots, binding.template
    if template == concept_list.ID:
        models = _section_models(binding.section)
        return [Target(template, slots[concept_list.RELATION], models, {})]
    if template == concept_with_values_denormalized.ID:
        return _denormalized(binding)
    if template == concept_and_value_set.ID:
        shape = concept_and_value_set
        return [
            Target(template, slots[shape.CONCEPT_RELATION], (ConceptRoles,),
                   {KEY_ROLE: slots[shape.CONCEPT_KEY]}),
            Target(template, slots[shape.VALUE_RELATION], (ValueSetRoles,),
                   {KEY_ROLE: slots[shape.VALUE_KEY]}),
        ]
    if template != entity_model.ID:
        return []
    if is_dictionary_binding(binding):
        return [Target(template, slots[entity_model.DICTIONARY], (EntityModelRoles,), {})]
    slotted = {
        entity_model.ENTITY: slots[entity_model.ENTITY],
        lexicon.ENTITY_PROPERTY_ROLE: slots[entity_model.PROPERTY],
    }
    return [Target(template, slots[entity_model.RELATION], (EntityModelRoles,), slotted)]


def side_profile(
    profile: RelationProfile, target: Target, tools: ProfileTools | None
) -> RelationProfile:
    """The columns a split target may bind; with no tools to test, every column.

    A constant column is determined by every key, so it may serve either side.
    """
    key = target.slotted.get(KEY_ROLE)
    if target.split is None or key is None or tools is None:
        return profile
    template = concept_with_values_denormalized
    wanted = target.split == "group"
    kept = tuple(
        column
        for column in profile.columns
        if column.name == key or column.is_constant
        or template.determines(profile, key, column.name, tools) == wanted
    )
    return replace(profile, columns=kept)


def memory_key(target: Target, profile: RelationProfile) -> MemoryKey:
    return scope_of(target.template, target.section), profile.signature


def _denormalized(binding: SlotBinding) -> list[Target]:
    template = concept_with_values_denormalized
    relation = binding.slots[template.RELATION]
    slotted = {KEY_ROLE: binding.slots[template.GROUP_BY]}
    return [
        Target(binding.template, relation, (ConceptRoles,), dict(slotted), split="group"),
        Target(
            binding.template, relation, (ValueSetRoles,), dict(slotted),
            section=VALUE_SETS, split="rows",
        ),
    ]


def _section_models(section: str | None) -> tuple[type[BaseModel], ...]:
    head = (section or "").partition(SECTION_SEPARATOR)[0]
    if head == CONCEPTS:
        return (ConceptRoles,)
    if head == CODE_LISTS:
        return (CodeListRoles,)
    return (ConceptRoles, CodeListRoles)
