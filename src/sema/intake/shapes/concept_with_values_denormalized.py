"""concept_with_values_denormalized: one relation grouped by a column, codes unique per group."""

from __future__ import annotations

from collections.abc import Sequence

from sema.binding.spec_sections import ConceptsSpec
from sema.intake.profile import ProfileTools
from sema.intake.profile_utils import ColumnProfile
from sema.intake.shapes.base import RelationProfile, Slot, SlotBinding
from sema.intake.shapes.base_utils import (
    failure,
    section_of,
    slot_columns,
    slot_relations,
    verdict,
)
from sema.intake.verify.base import Failure, FloorValue, Verdict

ID = "concept_with_values_denormalized"
SECTION = section_of(ConceptsSpec)
RELATION, GROUP_BY = "relation", "group_by"
SLOTS = (Slot(RELATION, "relation"), Slot(GROUP_BY, "column"))
# Measured: syn-library data_element.data_element_id holds 4 distinct values over 9 rows
# (0.444); every other syn and A2 id column is 1.0.
DENORMALIZED_GROUP_RATIO_FLOOR = 0.5
FLOORS: dict[str, FloorValue] = {"DENORMALIZED_GROUP_RATIO_FLOOR": DENORMALIZED_GROUP_RATIO_FLOOR}
MIN_GROUPS = 2


def candidates(group: Sequence[RelationProfile], tools: ProfileTools) -> list[SlotBinding]:
    found = []
    for profile in group:
        for column in profile.columns:
            if not _failures(profile, column, tools):
                found.append(_binding(profile, column))
    return found


def verify(
    binding: SlotBinding, group: Sequence[RelationProfile], tools: ProfileTools
) -> Verdict:
    found, failures = slot_relations(group, binding, (RELATION,))
    for profile in found:
        columns, missing = slot_columns(profile, binding, (GROUP_BY,))
        failures += missing + [f for c in columns for f in _failures(profile, c, tools)]
    return verdict(failures, FLOORS)


def _failures(
    profile: RelationProfile, group_by: ColumnProfile, tools: ProfileTools
) -> list[Failure]:
    relation = profile.ref.name
    if profile.row_count == 0 or group_by.null_rate > 0.0:
        return [failure("group_by_non_null", relation, group_by.name)]
    if group_by.distinct_count < MIN_GROUPS:
        return [failure("group_by_groups", relation, group_by.name, groups=MIN_GROUPS)]
    if group_by.distinct_ratio > DENORMALIZED_GROUP_RATIO_FLOOR:
        ratio = group_by.distinct_ratio
        return [failure("group_by_ratio", relation, group_by.name, distinct_ratio=ratio)]
    codes = [c for c in profile.columns if _unique_per_group(profile, group_by, c, tools)]
    if not codes:
        return [failure("code_unique_per_group", relation, group_by.name)]
    if not _is_entity(profile, group_by, codes, tools):
        return [failure("group_is_entity", relation, group_by.name)]
    return []


def _is_entity(
    profile: RelationProfile,
    group_by: ColumnProfile,
    codes: Sequence[ColumnProfile],
    tools: ProfileTools,
) -> bool:
    """A group scopes its codes (a weak key) or carries an attribute of its own."""
    if any(not code.is_unique for code in codes):
        return True
    return any(
        determines(profile, group_by.name, column.name, tools)
        for column in profile.columns
        if column.name != group_by.name and column.distinct_count > 1
    )


def determines(profile: RelationProfile, group_by: str, column: str, tools: ProfileTools) -> bool:
    """``group_by`` functionally determines ``column`` over the rows where it is non-null."""
    ref = profile.ref
    present = f"{tools.provider.quote(ref, column)} IS NOT NULL"
    groups = tools.count_distinct(ref, [group_by], where=present)
    return tools.count_distinct(ref, [group_by, column]) == groups


def _unique_per_group(
    profile: RelationProfile, group_by: ColumnProfile, code: ColumnProfile, tools: ProfileTools
) -> bool:
    if code.name == group_by.name or code.null_rate > 0.0:
        return False
    if code.is_unique:
        return True
    return tools.count_distinct(profile.ref, [group_by.name, code.name]) == profile.row_count


def _binding(profile: RelationProfile, group_by: ColumnProfile) -> SlotBinding:
    ref_name = profile.ref.name
    return SlotBinding(
        template=ID,
        slots={RELATION: ref_name, GROUP_BY: group_by.name},
        evidence={"groups": float(group_by.distinct_count)},
        relations=(ref_name,),
    )
