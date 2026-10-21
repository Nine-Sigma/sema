"""Helpers every template shares: binding lookups, failures, verdicts, join pre-checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sema.intake.profile_utils import ColumnProfile
from sema.intake.shapes.base import (
    ID_DISTINCT_RATIO_FLOOR,
    JOIN_RESOLVE_FLOOR,
    LABEL_MEAN_LENGTH_BOUNDS,
    RelationProfile,
    SlotBinding,
)
from sema.intake.verify.base import Failure, FloorValue, Verdict

ID_FLOOR: dict[str, FloorValue] = {"ID_DISTINCT_RATIO_FLOOR": ID_DISTINCT_RATIO_FLOOR}
JOIN_FLOOR: dict[str, FloorValue] = {"JOIN_RESOLVE_FLOOR": JOIN_RESOLVE_FLOOR}
SIGNATURE_FLOORS: dict[str, FloorValue] = {
    **ID_FLOOR,
    "LABEL_MEAN_LENGTH_BOUNDS": LABEL_MEAN_LENGTH_BOUNDS,
}
IN_GROUP = "in_group"
COLUMN_EXISTS = "column_exists"


def failure(check: str, relation: str, column: str | None = None, **evidence: Any) -> Failure:
    return Failure(check=check, relation=relation, column=column, evidence=evidence)


def verdict(failures: Sequence[Failure], floors: Mapping[str, FloorValue]) -> Verdict:
    return Verdict.of(failures=list(failures), ambiguities=[], floors=dict(floors))


def find(group: Sequence[RelationProfile], ref_name: str | None) -> RelationProfile | None:
    return next((p for p in group if p.ref.name == ref_name), None)


def slot_relations(
    group: Sequence[RelationProfile], binding: SlotBinding, slots: Sequence[str]
) -> tuple[list[RelationProfile], list[Failure]]:
    found, failures = [], []
    for slot in slots:
        ref_name = binding.slots.get(slot)
        profile = find(group, ref_name)
        if profile is None:
            failures.append(failure(IN_GROUP, ref_name or "", slot=slot))
        else:
            found.append(profile)
    return found, failures


def slot_columns(
    profile: RelationProfile, binding: SlotBinding, slots: Sequence[str]
) -> tuple[list[ColumnProfile], list[Failure]]:
    found, failures = [], []
    for slot in slots:
        name = binding.slots.get(slot)
        column = profile.column(name) if name is not None else None
        if column is None:
            failures.append(failure(COLUMN_EXISTS, profile.ref.name, name, slot=slot))
        else:
            found.append(column)
    return found, failures


def may_resolve(column: ColumnProfile, target: ColumnProfile) -> bool:
    """Necessary for ``column`` to resolve into ``target`` above the join floor."""
    return (
        column.type == target.type
        and column.distinct_count > 0
        and column.distinct_count * JOIN_RESOLVE_FLOOR <= target.distinct_count
    )


def nonnull_rows(profile: RelationProfile, column: ColumnProfile) -> int:
    return round(profile.row_count * (1.0 - column.null_rate))


def fixed_relation_sets(
    group: Sequence[RelationProfile], required: Sequence[str]
) -> list[dict[str, RelationProfile]]:
    """Per namespace, the loader-fixed relations by name, when every required one is landed."""
    namespaces: dict[str, dict[str, RelationProfile]] = {}
    for profile in group:
        namespaces.setdefault(profile.landed.schema, {})[profile.landed.name] = profile
    return [
        {name: landed[name] for name in required}
        for landed in namespaces.values()
        if all(name in landed for name in required)
    ]


def fixed_binding(template: str, relations: dict[str, RelationProfile]) -> SlotBinding:
    slots = {name: profile.ref.name for name, profile in relations.items()}
    return SlotBinding(template=template, slots=slots, relations=tuple(slots.values()))


def fixed_failures(
    group: Sequence[RelationProfile], binding: SlotBinding, required: Sequence[str]
) -> list[Failure]:
    """The binding names each required relation, and each is in the group under that name."""
    failures = []
    for name in required:
        profile = find(group, binding.slots.get(name))
        if profile is None or profile.landed.name != name:
            failures.append(failure(IN_GROUP, binding.slots.get(name, ""), slot=name))
    return failures
