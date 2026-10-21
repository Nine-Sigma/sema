"""entity_model: per_file dictionaries with a key-value header, or one relation.

``per_file``: in a stem group, a dictionary (a unique non-null text column beside another
text column) and an optional header (two columns, unique non-null first). A dictionary with no
header in its group is indistinguishable from a code list, so its header-less binding stays a
candidate beside ``concept_list`` and precedence leaves the relation ambiguous. A key-value
relation in a group that holds another dictionary is a header only. ``single_relation``: an
entity column and a property column unique within each entity whose global ratio stays below
the id floor, because property names recur across entities while codes do not.
"""

from __future__ import annotations

from collections.abc import Sequence

from sema.binding.spec import EntityModelSpec
from sema.intake.profile import ProfileTools
from sema.intake.profile_utils import ColumnProfile
from sema.intake.shapes.base import (
    ID_DISTINCT_RATIO_FLOOR,
    RelationProfile,
    SlotBinding,
    is_id,
    is_text,
)
from sema.intake.shapes.base_utils import (
    ID_FLOOR,
    failure,
    section_of,
    slot_columns,
    slot_relations,
    verdict,
)
from sema.intake.verify.base import Failure, FloorValue, Verdict

ID = "entity_model"
SECTION = section_of(EntityModelSpec)
LAYOUT, PER_FILE, SINGLE_RELATION = "layout", "per_file", "single_relation"
DICTIONARY, HEADER = "dictionary", "header"
RELATION, ENTITY, PROPERTY = "relation", "entity", "property"
# Measured: OMOP field-level CSV cdmTableName 39 distinct over 432 rows (0.09), while its
# property column cdmFieldName is 0.771.
ENTITY_DISTINCT_RATIO_CAP = 0.5
FLOORS: dict[str, FloorValue] = {**ID_FLOOR, "ENTITY_DISTINCT_RATIO_CAP": ENTITY_DISTINCT_RATIO_CAP}
KEY_VALUE_WIDTH = 2
MIN_ENTITIES = 2


def candidates(group: Sequence[RelationProfile], tools: ProfileTools) -> list[SlotBinding]:
    found = _per_file_bindings(group)
    for profile in group:
        found += _single_bindings(profile, tools)
    return found


def verify(
    binding: SlotBinding, group: Sequence[RelationProfile], tools: ProfileTools
) -> Verdict:
    if binding.slots.get(LAYOUT) == PER_FILE:
        return verdict(_per_file_failures(binding, group), FLOORS)
    if binding.slots.get(LAYOUT) != SINGLE_RELATION:
        relation = binding.relations[0] if binding.relations else ""
        return verdict([failure(LAYOUT, relation, layout=binding.slots.get(LAYOUT, ""))], FLOORS)
    found, failures = slot_relations(group, binding, (RELATION,))
    for profile in found:
        columns, missing = slot_columns(profile, binding, (ENTITY, PROPERTY))
        failures += missing
        if not missing:
            entity, prop = columns
            failures += _entity_failures(profile, entity)
            failures += _property_failures(profile, entity, prop, tools)
    return verdict(failures, FLOORS)


def is_dictionary(profile: RelationProfile) -> bool:
    keys = [c for c in profile.columns if is_id(profile, c) and is_text(c)]
    texts = [c for c in profile.columns if is_text(c) and c.mean_length is not None]
    return any(other.name != key.name for key in keys for other in texts)


def is_key_value(profile: RelationProfile) -> bool:
    return len(profile.columns) == KEY_VALUE_WIDTH and is_id(profile, profile.columns[0])


def header_only(profile: RelationProfile, group: Sequence[RelationProfile]) -> bool:
    return is_key_value(profile) and any(
        is_dictionary(other) for other in group if other.ref.name != profile.ref.name
    )


def _per_file_bindings(group: Sequence[RelationProfile]) -> list[SlotBinding]:
    headers = [p for p in group if is_key_value(p)]
    found = []
    for dictionary in (p for p in group if is_dictionary(p) and not header_only(p, group)):
        own = [header for header in headers if header.ref.name != dictionary.ref.name]
        found += [_per_file_binding(dictionary, header) for header in own]
        if not own:
            found.append(_per_file_binding(dictionary, None))
    return found


def _single_bindings(profile: RelationProfile, tools: ProfileTools) -> list[SlotBinding]:
    entities = [c for c in profile.columns if not _entity_failures(profile, c)]
    return [
        _single_binding(profile, entity, prop)
        for entity in entities
        for prop in profile.columns
        if not _property_failures(profile, entity, prop, tools)
    ]


def _per_file_failures(binding: SlotBinding, group: Sequence[RelationProfile]) -> list[Failure]:
    slots = (DICTIONARY, HEADER) if HEADER in binding.slots else (DICTIONARY,)
    found, failures = slot_relations(group, binding, slots)
    if failures:
        return failures
    dictionary, *headers = found
    if not is_dictionary(dictionary):
        failures.append(failure("dictionary_signature", dictionary.ref.name))
    if header_only(dictionary, group):
        failures.append(failure("header_only", dictionary.ref.name))
    for header in headers:
        failures += _header_failures(dictionary, header)
    return failures


def _header_failures(dictionary: RelationProfile, header: RelationProfile) -> list[Failure]:
    failures = []
    if not is_key_value(header):
        failures.append(failure("header_key_value", header.ref.name, width=len(header.columns)))
    if header is dictionary:
        failures.append(failure("header_is_another_relation", header.ref.name))
    return failures


def _entity_failures(profile: RelationProfile, entity: ColumnProfile) -> list[Failure]:
    relation = profile.ref.name
    if not is_text(entity) or entity.null_rate > 0.0 or profile.row_count == 0:
        return [failure("entity_text_non_null", relation, entity.name)]
    if entity.distinct_count < MIN_ENTITIES:
        return [failure("entity_count", relation, entity.name, entities=entity.distinct_count)]
    if entity.distinct_ratio > ENTITY_DISTINCT_RATIO_CAP:
        ratio = entity.distinct_ratio
        return [failure("entity_ratio", relation, entity.name, distinct_ratio=ratio)]
    return []


def _property_failures(
    profile: RelationProfile, entity: ColumnProfile, prop: ColumnProfile, tools: ProfileTools
) -> list[Failure]:
    relation = profile.ref.name
    if prop.name == entity.name or not is_text(prop) or prop.null_rate > 0.0:
        return [failure("property_text_non_null", relation, prop.name)]
    if prop.distinct_ratio >= ID_DISTINCT_RATIO_FLOOR:
        return [failure("property_recurs", relation, prop.name, ratio=prop.distinct_ratio)]
    pairs = tools.count_distinct(profile.ref, [entity.name, prop.name])
    if pairs != profile.row_count:
        return [failure("property_unique_per_entity", relation, prop.name, pairs=pairs)]
    return []


def _per_file_binding(
    dictionary: RelationProfile, header: RelationProfile | None
) -> SlotBinding:
    slots = {LAYOUT: PER_FILE, DICTIONARY: dictionary.ref.name}
    relations = [dictionary.ref.name]
    if header is not None:
        slots[HEADER] = header.ref.name
        relations.append(header.ref.name)
    return SlotBinding(template=ID, slots=slots, relations=tuple(relations))


def _single_binding(
    profile: RelationProfile, entity: ColumnProfile, prop: ColumnProfile
) -> SlotBinding:
    ref_name = profile.ref.name
    return SlotBinding(
        template=ID,
        slots={
            LAYOUT: SINGLE_RELATION,
            RELATION: ref_name,
            ENTITY: entity.name,
            PROPERTY: prop.name,
        },
        relations=(ref_name,),
    )
