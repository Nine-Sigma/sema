"""Described data is not a dictionary (relation-structure).

A relation whose column names equal, after the lexicon's name normalization, the property
names a dictionary lists is the data that dictionary describes, so no ``entity_model``
binding may use it as a dictionary or header. A dictionary's property names are the values
of one of its key columns (unique, non-null text). They are compared inside SQL, and only
counts come back, so the comparison issues no value-bearing statement and reads no text
before binding. A key column is compared only when it holds as many values as some target
relation has columns.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence

from sema.intake.landing import LOADER_COLUMN_PREFIX
from sema.intake.lexicon import normalize
from sema.intake.profile import ProfileTools
from sema.intake.profile_utils import ColumnProfile
from sema.intake.shapes import entity_model
from sema.intake.shapes.base import RelationProfile, SlotBinding, is_id, is_text
from sema.log import logger

SCHEMA_SLOTS = (entity_model.DICTIONARY, entity_model.HEADER, entity_model.RELATION)

Names = frozenset[str]


def per_file_dictionaries(bindings: Iterable[SlotBinding]) -> list[str]:
    return list(
        dict.fromkeys(
            binding.slots[entity_model.DICTIONARY]
            for binding in bindings
            if binding.template == entity_model.ID
            and binding.slots.get(entity_model.LAYOUT) == entity_model.PER_FILE
        )
    )


def schema_slot_relations(bindings: Iterable[SlotBinding]) -> list[str]:
    return list(
        dict.fromkeys(
            binding.slots[slot]
            for binding in bindings
            if binding.template == entity_model.ID
            for slot in SCHEMA_SLOTS
            if slot in binding.slots
        )
    )


def uses_described(binding: SlotBinding, described: Collection[str]) -> bool:
    return binding.template == entity_model.ID and any(
        binding.slots.get(slot) in described for slot in SCHEMA_SLOTS
    )


def described_by(
    dictionaries: Sequence[RelationProfile],
    targets: Sequence[RelationProfile],
    tools: ProfileTools,
) -> dict[str, str]:
    """Each target whose column names a dictionary lists, mapped to the first such dictionary."""
    keys = [(d, key) for d in dictionaries for key in d.columns if _is_key(d, key)]
    found: dict[str, str] = {}
    for target in targets:
        by = next((d for d, key in keys if _describes(tools, d, key, target)), None)
        if by is not None:
            found[target.ref.name] = by.ref.name
    if found:
        logger.info("described data, not dictionaries: {}", sorted(found))
    return found


def _landed_columns(profile: RelationProfile) -> list[str]:
    return [c.name for c in profile.columns if not c.name.startswith(LOADER_COLUMN_PREFIX)]


def _column_names(profile: RelationProfile) -> Names:
    return frozenset(map(normalize, _landed_columns(profile)))


def _is_key(profile: RelationProfile, column: ColumnProfile) -> bool:
    return is_id(profile, column) and is_text(column)


def _describes(
    tools: ProfileTools, dictionary: RelationProfile, key: ColumnProfile, target: RelationProfile
) -> bool:
    """Whether ``key`` lists exactly the target's column names, compared by counts only."""
    names = _column_names(target)
    if not names or dictionary.ref.name == target.ref.name:
        return False
    if key.distinct_count != len(_landed_columns(target)):
        return False
    return tools.count_listed_names(dictionary.ref, key.name, names).lists_exactly(len(names))
