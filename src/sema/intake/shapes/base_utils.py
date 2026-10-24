"""Helpers every template shares: binding lookups, failures, verdicts, join pre-checks."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, get_args

from sema.binding.spec import ConceptSourceSpec
from sema.intake.profile import ProfileTools
from sema.intake.profile_utils import ColumnProfile
from sema.intake.shapes.base import (
    ID_DISTINCT_RATIO_FLOOR,
    JOIN_RESOLVE_FLOOR,
    LABEL_MEAN_LENGTH_BOUNDS,
    RelationProfile,
    SlotBinding,
)
from sema.intake.verify.base import Failure, Finding, FloorClass, FloorValue, Verdict

ID_FLOOR: dict[str, FloorValue] = {"ID_DISTINCT_RATIO_FLOOR": ID_DISTINCT_RATIO_FLOOR}
JOIN_FLOOR: dict[str, FloorValue] = {"JOIN_RESOLVE_FLOOR": JOIN_RESOLVE_FLOOR}
SIGNATURE_FLOORS: dict[str, FloorValue] = {
    **ID_FLOOR,
    "LABEL_MEAN_LENGTH_BOUNDS": LABEL_MEAN_LENGTH_BOUNDS,
}
# Numeric and temporal types order the same way on every provider. Text is left out: its
# order follows a column's collation, which may differ between two columns of one type.
ORDERED_TYPE_WORDS = frozenset({
    "TINYINT", "SMALLINT", "INT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
    "UINTEGER", "UBIGINT", "BYTE", "SHORT", "LONG", "FLOAT", "DOUBLE", "REAL", "DECIMAL",
    "NUMERIC", "DATE", "TIMESTAMP", "TIMESTAMPTZ",
})
TYPE_WORD = re.compile(r"[A-Z]+")
ColumnPair = tuple[ColumnProfile, ColumnProfile]
IN_GROUP = "in_group"
COLUMN_EXISTS = "column_exists"


def section_of(model: type) -> str:
    """The spec section whose field holds ``model``, so no template spells a section name."""
    return next(
        name
        for name, field in ConceptSourceSpec.model_fields.items()
        if model is field.annotation or model in get_args(field.annotation)
    )


def named_section(name: str) -> str:
    """A section whose model several sections share, so it is named and checked here."""
    if name not in ConceptSourceSpec.model_fields:
        raise ValueError(f"{name} is not a section of the concept source spec")
    return name


def failure(check: str, relation: str, column: str | None = None, **evidence: Any) -> Failure:
    return Failure(check=check, relation=relation, column=column, evidence=evidence)


def finding(name: str, relation: str, column: str | None, value: float | str | None) -> Finding:
    subject = relation if column is None else f"{relation}.{column}"
    return Finding(name=name, subject=subject, value=value)


def verdict(
    failures: Sequence[Failure],
    floors: Mapping[str, FloorValue],
    classes: Mapping[str, FloorClass] | None = None,
    findings: Sequence[Finding] = (),
) -> Verdict:
    """Every listed constant carries its class; an unclassed one is a programming error."""
    classes = classes or {}
    unclassed = sorted(set(floors) - set(classes))
    if unclassed:
        raise ValueError(f"constants listed without a class: {unclassed}")
    return Verdict.of(
        failures=list(failures),
        ambiguities=[],
        floors=dict(floors),
        floor_classes={name: classes[name] for name in floors},
        findings=list(findings),
    )


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


def ranges_overlapping(
    profile: RelationProfile, pairs: Sequence[ColumnPair], tools: ProfileTools
) -> list[ColumnPair]:
    """``pairs`` less those of an ordered type whose value ranges do not overlap.

    A column resolves into another at the join floor only if they share a value, and a
    shared value lies inside both ranges, so a disjoint pair cannot resolve.
    """
    groups: dict[str, list[str]] = {}
    for column in dict.fromkeys(c for pair in pairs if _ordered(pair[0]) for c in pair):
        groups.setdefault(column.type, []).append(column.name)
    if not groups:
        return list(pairs)
    overlapping = tools.overlapping_ranges(profile.ref, list(groups.values()))
    return [
        pair
        for pair in pairs
        if not _ordered(pair[0]) or frozenset((pair[0].name, pair[1].name)) in overlapping
    ]


def _ordered(column: ColumnProfile) -> bool:
    word = TYPE_WORD.match(column.type.upper())
    return word is not None and word.group() in ORDERED_TYPE_WORDS


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


def fixed_binding(
    template: str, relations: dict[str, RelationProfile], group: Sequence[RelationProfile]
) -> SlotBinding:
    """The binding spans every relation of the namespace, because its loader landed them all."""
    slots = {name: profile.ref.name for name, profile in relations.items()}
    schemas = {profile.landed.schema for profile in relations.values()}
    landed = [profile.ref.name for profile in group if profile.landed.schema in schemas]
    spanned = tuple(dict.fromkeys([*slots.values(), *landed]))
    return SlotBinding(template=template, slots=slots, relations=spanned)


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
