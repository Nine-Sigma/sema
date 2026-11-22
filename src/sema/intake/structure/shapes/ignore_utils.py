"""Boilerplate detection for the ignore template: a relation repeated unchanged across files."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

from sema.intake.land.profile import ProfileTools
from sema.intake.structure.shapes.base import RelationProfile

FULL_RESOLVE = 1.0


def stem_groups(group: Sequence[RelationProfile]) -> dict[str, list[RelationProfile]]:
    groups: dict[str, list[RelationProfile]] = {}
    for profile in group:
        groups.setdefault(profile.stem, []).append(profile)
    return groups


def other_multi_groups(
    profile: RelationProfile, group: Sequence[RelationProfile]
) -> list[list[RelationProfile]] | None:
    """The other multi-relation stem groups, or None when ``profile`` is not in one."""
    groups = stem_groups(group)
    if len(groups.get(profile.stem, ())) < 2:
        return None
    return [
        members
        for stem, members in groups.items()
        if stem != profile.stem and len(members) > 1
    ]


def repeated_share(
    profile: RelationProfile,
    others: Sequence[Sequence[RelationProfile]],
    floor: float,
    tools: ProfileTools,
) -> float | None:
    """The share of ``others`` holding a relation identical to ``profile``, or None below
    ``floor``.

    Stops as soon as the share cannot reach the floor, so a sheet unique to its file costs
    few joins; a share at the floor is counted in full, because the binding states it.
    """
    if not others:
        return None
    needed = ceil(floor * len(others))
    matched = 0
    for checked, members in enumerate(others, 1):
        matched += any(identical(profile, other, tools) for other in members)
        if matched + len(others) - checked < needed:
            return None
    return matched / len(others)


def identical(a: RelationProfile, b: RelationProfile, tools: ProfileTools) -> bool:
    """Same signature, same row count, and every column's distinct values equal both ways."""
    if a.ref.provider != b.ref.provider:
        return False
    if a.signature != b.signature or a.row_count != b.row_count:
        return False
    for column in a.columns:
        twin = b.column(column.name)
        if twin is None or twin.distinct_count != column.distinct_count:
            return False
        if not column.distinct_count:
            continue
        joined = tools.check_join(a.ref, column.name, b.ref, column.name)
        if joined.resolve_a_to_b < FULL_RESOLVE or joined.resolve_b_to_a < FULL_RESOLVE:
            return False
    return True
