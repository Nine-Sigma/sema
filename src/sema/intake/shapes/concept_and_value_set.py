"""concept_and_value_set: a concept relation, a value relation, and a resolving key pair."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import product

from sema.intake.profile import ProfileTools
from sema.intake.profile_utils import ColumnProfile
from sema.intake.shapes.base import JOIN_RESOLVE_FLOOR, RelationProfile, SlotBinding, is_id
from sema.intake.shapes.base_utils import (
    JOIN_FLOOR,
    failure,
    may_resolve,
    slot_columns,
    slot_relations,
    verdict,
)
from sema.intake.verify.base import Failure, Verdict

ID = "concept_and_value_set"
CONCEPT_RELATION, VALUE_RELATION = "concept_relation", "value_relation"
CONCEPT_KEY, VALUE_KEY = "concept_key", "value_key"
ONE_TO_MANY = "one_to_many"

KeyPair = tuple[RelationProfile, ColumnProfile, RelationProfile, ColumnProfile]


def candidates(group: Sequence[RelationProfile], tools: ProfileTools) -> list[SlotBinding]:
    return [_binding(*pair) for pair in _key_pairs(group) if not _join_failures(*pair, tools)]


def verify(
    binding: SlotBinding, group: Sequence[RelationProfile], tools: ProfileTools
) -> Verdict:
    found, failures = slot_relations(group, binding, (CONCEPT_RELATION, VALUE_RELATION))
    if failures:
        return verdict(failures, JOIN_FLOOR)
    concepts, values = found
    if concepts.ref.name == values.ref.name:
        return verdict([failure("distinct_relations", values.ref.name)], JOIN_FLOOR)
    keys, failures = slot_columns(concepts, binding, (CONCEPT_KEY,))
    value_keys, missing = slot_columns(values, binding, (VALUE_KEY,))
    if failures or missing:
        return verdict(failures + missing, JOIN_FLOOR)
    key, value_key = keys[0], value_keys[0]
    if not is_id(concepts, key):
        failures.append(failure("concept_key_is_id", concepts.ref.name, key.name))
    failures += _join_failures(concepts, key, values, value_key, tools)
    return verdict(failures, JOIN_FLOOR)


def _key_pairs(group: Sequence[RelationProfile]) -> Iterator[KeyPair]:
    for concepts in group:
        keys = [column for column in concepts.columns if is_id(concepts, column)]
        for values in (p for p in group if p.ref.name != concepts.ref.name):
            for key, value_key in product(keys, values.columns):
                if _repeats(value_key) and may_resolve(value_key, key):
                    yield concepts, key, values, value_key


def _repeats(column: ColumnProfile) -> bool:
    return 0.0 < column.distinct_ratio < 1.0


def _join_failures(
    concepts: RelationProfile,
    key: ColumnProfile,
    values: RelationProfile,
    value_key: ColumnProfile,
    tools: ProfileTools,
) -> list[Failure]:
    relation = values.ref.name
    if value_key.type != key.type:
        return [failure("key_types_match", relation, value_key.name, concept_type=key.type)]
    joined = tools.check_join(concepts.ref, key.name, values.ref, value_key.name)
    failures = []
    if joined.resolve_b_to_a < JOIN_RESOLVE_FLOOR:
        failures.append(
            failure("join_resolves", relation, value_key.name, resolve=joined.resolve_b_to_a)
        )
    if joined.cardinality != ONE_TO_MANY:
        failures.append(
            failure("one_to_many", relation, value_key.name, cardinality=joined.cardinality)
        )
    return failures


def _binding(
    concepts: RelationProfile, key: ColumnProfile, values: RelationProfile, value_key: ColumnProfile
) -> SlotBinding:
    concept_ref, value_ref = concepts.ref.name, values.ref.name
    return SlotBinding(
        template=ID,
        slots={
            CONCEPT_RELATION: concept_ref,
            VALUE_RELATION: value_ref,
            CONCEPT_KEY: key.name,
            VALUE_KEY: value_key.name,
        },
        relations=(concept_ref, value_ref),
    )
