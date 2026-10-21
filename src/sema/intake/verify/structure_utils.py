"""Structure verifier helpers: partition counts, binding identity, the boilerplate re-check."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from sema.intake.profile import ProfileTools
from sema.intake.shapes.base import RelationProfile, SlotBinding
from sema.intake.shapes.ignore_utils import FULL_RESOLVE, other_multi_groups
from sema.intake.structure import StructureResult

BindingKey = tuple[str, tuple[tuple[str, str], ...], tuple[str, ...]]


def partition_counts(proposal: StructureResult) -> Counter[str]:
    """How often each relation appears across bound groups, ignores, and open ambiguities.

    A settled ambiguity stays on the list as the record of its choice; the binding it chose
    is what covers the relation.
    """
    counts: Counter[str] = Counter()
    for entry in proposal.bound:
        counts.update(set(entry.binding.relations))
    for binding in proposal.ignored:
        counts.update(set(binding.relations))
    counts.update(a.subject for a in proposal.ambiguities if a.choice is None)
    return counts


def binding_key(binding: SlotBinding) -> BindingKey:
    """What the predicates decide; the section and the evidence are set elsewhere."""
    return binding.template, tuple(sorted(binding.slots.items())), binding.relations


def boilerplate_column(
    profile: RelationProfile, profiles: Sequence[RelationProfile], tools: ProfileTools
) -> str | None:
    """The first column that differs from a same-signature counterpart in another group."""
    for members in other_multi_groups(profile, profiles) or ():
        for other in members:
            if other.signature != profile.signature:
                continue
            column = differing_column(profile, other, tools)
            if column is not None:
                return column
    return None


def differing_column(
    a: RelationProfile, b: RelationProfile, tools: ProfileTools
) -> str | None:
    for column in a.columns:
        twin = b.column(column.name)
        if twin is None or twin.distinct_count != column.distinct_count:
            return column.name
        if not column.distinct_count:
            continue
        joined = tools.check_join(a.ref, column.name, b.ref, column.name)
        if min(joined.resolve_a_to_b, joined.resolve_b_to_a) < FULL_RESOLVE:
            return column.name
    return None
