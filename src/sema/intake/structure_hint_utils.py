"""Hint pinning: a partial spec's shape entries become slot bindings a human stated.

Each builder reads one spec section and fills the template's own slots, so a pinned binding
goes through the same ``verify`` as a code-found one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import (
    CodeListRoles,
    CodeListSpec,
    ConceptsSpec,
    EntityModelSpec,
    LevelColumnsHierarchy,
    ParentColumnHierarchy,
    PathCodesHierarchy,
    RelationRef,
    ValueSetRoles,
)
from sema.intake.shapes import (
    concept_and_value_set,
    concept_list,
    concept_with_values_denormalized,
    edge_list_hierarchy,
    entity_model,
    level_columns,
    path_codes,
)
from sema.intake.shapes.base import SlotBinding
from sema.intake.shapes.base_utils import failure
from sema.intake.verify.base import Failure

SECTION_SEPARATOR = "."
HINT_SECTION = "hint_section"
# The one role a value set adds over a code list: the key tying each value to its concept.
(KEY_ROLE,) = sorted(set(ValueSetRoles.model_fields) - set(CodeListRoles.model_fields))

Builder = Callable[[ConceptSourceSpec, object], list[SlotBinding]]


def hint_bindings(hint: ConceptSourceSpec) -> tuple[list[SlotBinding], list[Failure]]:
    bindings: list[SlotBinding] = []
    failures: list[Failure] = []
    for entry in hint.shape or ():
        builder = BUILDERS.get(entry.template)
        section = _section(hint, entry.section)
        built = builder(hint, section) if builder is not None and section is not None else []
        if not built:
            failures.append(
                failure(HINT_SECTION, "", section=entry.section, template=entry.template)
            )
        bindings += [binding.model_copy(update={"section": entry.section}) for binding in built]
    return bindings, failures


def _section(hint: ConceptSourceSpec, path: str) -> object | None:
    head, _, tail = path.partition(SECTION_SEPARATOR)
    value = getattr(hint, head) if head in type(hint).model_fields else None
    if not tail:
        return value
    if not isinstance(value, list):
        return None
    return next(
        (item for item in value if isinstance(item, CodeListSpec) and item.vocabulary_id == tail),
        None,
    )


def _binding(template: str, slots: Mapping[str, str], *relations: RelationRef) -> SlotBinding:
    spanned = tuple(dict.fromkeys(relation.name for relation in relations))
    return SlotBinding(template=template, slots=dict(slots), relations=spanned)


def _concept_list(hint: ConceptSourceSpec, section: object) -> list[SlotBinding]:
    if not isinstance(section, CodeListSpec | ConceptsSpec):
        return []
    relation = section.relation
    return [_binding(concept_list.ID, {concept_list.RELATION: relation.name}, relation)]


def _denormalized(hint: ConceptSourceSpec, section: object) -> list[SlotBinding]:
    if not isinstance(section, ConceptsSpec):
        return []
    template = concept_with_values_denormalized
    slots = {template.RELATION: section.relation.name, template.GROUP_BY: section.group_by}
    return [_binding(template.ID, slots, section.relation)]


def _concept_and_value_set(hint: ConceptSourceSpec, section: object) -> list[SlotBinding]:
    concepts, values = hint.concepts, hint.value_sets
    if not isinstance(section, ConceptsSpec) or concepts is None or values is None:
        return []
    template = concept_and_value_set
    slots = {
        template.CONCEPT_RELATION: concepts.relation.name,
        template.VALUE_RELATION: values.relation.name,
        template.CONCEPT_KEY: getattr(concepts.roles, KEY_ROLE),
        template.VALUE_KEY: getattr(values.roles, KEY_ROLE),
    }
    return [_binding(template.ID, slots, concepts.relation, values.relation)]


def _hierarchy_relation(hint: ConceptSourceSpec, own: RelationRef | None) -> RelationRef | None:
    return own if own is not None else (hint.concepts.relation if hint.concepts else None)


def _level_columns(hint: ConceptSourceSpec, section: object) -> list[SlotBinding]:
    if not isinstance(section, LevelColumnsHierarchy):
        return []
    relation = _hierarchy_relation(hint, section.relation)
    if relation is None:
        return []
    levels = enumerate(section.level_columns, 1)
    slots = {level_columns.RELATION: relation.name}
    slots |= {f"{level_columns.LEVEL}{depth}": name for depth, name in levels}
    return [_binding(level_columns.ID, slots, relation)]


def _path_codes(hint: ConceptSourceSpec, section: object) -> list[SlotBinding]:
    if not isinstance(section, PathCodesHierarchy):
        return []
    relation = _hierarchy_relation(hint, section.relation)
    if relation is None:
        return []
    slots = {
        path_codes.RELATION: relation.name,
        path_codes.CODE: section.path_column,
        path_codes.SEPARATOR: section.separator,
    }
    return [_binding(path_codes.ID, slots, relation)]


def _edge_list(hint: ConceptSourceSpec, section: object) -> list[SlotBinding]:
    concepts = hint.concepts
    if not isinstance(section, ParentColumnHierarchy) or concepts is None:
        return []
    template = edge_list_hierarchy
    slots = {
        template.CONCEPT_RELATION: concepts.relation.name,
        template.IDENTIFIER: getattr(concepts.roles, KEY_ROLE),
        template.PARENT: section.parent_column,
    }
    edges = section.relation
    if edges is None or edges.name == concepts.relation.name:
        return [_binding(template.ID, slots, concepts.relation)]
    if section.child_column is None:
        return []
    slots |= {template.EDGE_RELATION: edges.name, template.CHILD: section.child_column}
    return [_binding(template.ID, slots, concepts.relation, edges)]


def _entity_model(hint: ConceptSourceSpec, section: object) -> list[SlotBinding]:
    if not isinstance(section, EntityModelSpec):
        return []
    template = entity_model
    if section.layout == template.SINGLE_RELATION:
        if section.relation is None or section.roles.entity is None:
            return []
        slots = {
            template.LAYOUT: template.SINGLE_RELATION,
            template.RELATION: section.relation.name,
            template.ENTITY: section.roles.entity,
            template.PROPERTY: section.roles.property,
        }
        return [_binding(template.ID, slots, section.relation)]
    found = []
    for entry in section.entities or ():
        slots = {template.LAYOUT: template.PER_FILE, template.DICTIONARY: entry.relation.name}
        headers = (entry.header,) if entry.header is not None else ()
        slots |= {template.HEADER: header.name for header in headers}
        found.append(_binding(template.ID, slots, entry.relation, *headers))
    return found


BUILDERS: Mapping[str, Builder] = {
    concept_list.ID: _concept_list,
    concept_with_values_denormalized.ID: _denormalized,
    concept_and_value_set.ID: _concept_and_value_set,
    level_columns.ID: _level_columns,
    path_codes.ID: _path_codes,
    edge_list_hierarchy.ID: _edge_list,
    entity_model.ID: _entity_model,
}
