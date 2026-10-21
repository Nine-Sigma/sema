"""Boilerplate detection for the ignore template: a relation repeated unchanged across files."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

from sema.intake.profile import ProfileTools
from sema.intake.shapes.base import RelationProfile

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


def repeated_in_share(
    profile: RelationProfile,
    others: Sequence[Sequence[RelationProfile]],
    floor: float,
    tools: ProfileTools,
) -> bool:
    """Whether at least ``floor`` of ``others`` hold a relation identical to ``profile``.

    Stops as soon as the outcome is decided, so a sheet unique to its file costs few joins.
    """
    if not others:
        return False
    needed = ceil(floor * len(others))
    matched = 0
    for checked, members in enumerate(others, 1):
        matched += any(identical(profile, other, tools) for other in members)
        if matched >= needed:
            return True
        if matched + len(others) - checked < needed:
            return False
    return False


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
