"""Described data is not a dictionary (relation-structure).

A relation whose column names equal, after the lexicon's name normalization, the property
names a dictionary lists is the data that dictionary describes, so no ``entity_model``
binding may use it as a dictionary or header. A dictionary's property names are the values
of one of its key columns (unique, non-null text), read as schema text. A dictionary is read
only when one of its key columns holds as many names as some target relation has columns,
so a dictionary that cannot list any target's column names is never read.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

from sema.binding.importer_utils import as_text
from sema.binding.spec_sections import RelationRef
from sema.intake.landing import LOADER_COLUMN_PREFIX
from sema.intake.lexicon import normalize
from sema.intake.profile import ProfileTools
from sema.intake.profile_utils import ColumnProfile
from sema.intake.shapes import entity_model
from sema.intake.shapes.base import RelationProfile, SlotBinding, is_id, is_text
from sema.log import logger

SCHEMA_SLOTS = (entity_model.DICTIONARY, entity_model.HEADER, entity_model.RELATION)

Names = frozenset[str]


@dataclass(frozen=True)
class SchemaScope:
    """The dictionaries a read may open while the structure output is still in progress."""

    schema_relations: frozenset[str]


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
    columns = {target.ref.name: _column_names(target) for target in targets}
    widths = {len(_landed_columns(target)) for target in targets}
    readable = {d.ref.name: keys for d in dictionaries if (keys := _key_columns(d, widths))}
    if not readable:
        return {}
    scoped = tools.scoped_to(SchemaScope(frozenset(readable)))
    by_name = {dictionary.ref.name: dictionary for dictionary in dictionaries}
    found: dict[str, str] = {}
    for name, keys in readable.items():
        lists = _property_lists(scoped, by_name[name].ref, keys)
        for target, names in columns.items():
            if target != name and names in lists:
                found.setdefault(target, name)
    if found:
        logger.info("described data, not dictionaries: {}", sorted(found))
    return found


def _landed_columns(profile: RelationProfile) -> list[str]:
    return [c.name for c in profile.columns if not c.name.startswith(LOADER_COLUMN_PREFIX)]


def _column_names(profile: RelationProfile) -> Names:
    return frozenset(map(normalize, _landed_columns(profile)))


def _key_columns(profile: RelationProfile, widths: Collection[int]) -> list[ColumnProfile]:
    return [
        column
        for column in profile.columns
        if is_id(profile, column) and is_text(column) and column.distinct_count in widths
    ]


def _property_lists(
    tools: ProfileTools, ref: RelationRef, keys: Sequence[ColumnProfile]
) -> list[Names]:
    rows = tools.read_schema_text(ref, [column.name for column in keys])
    return [
        frozenset(
            normalize(text)
            for row in rows
            if (text := as_text(row[index])) and not text.startswith(LOADER_COLUMN_PREFIX)
        )
        for index in range(len(keys))
    ]
