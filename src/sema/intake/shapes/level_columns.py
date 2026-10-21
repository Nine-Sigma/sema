"""level_columns: ordered columns where nulls sit only below and each prefix path is one node.

Under that rule a shallower level is never more null than a deeper one and, at equal null
rates, never has more distinct nodes, so the order is read from those aggregates. A level
that is neither constant nor unique is interior; without one the relation is a flat list.
"""

from __future__ import annotations

from collections.abc import Sequence

from sema.binding.spec_sections import LevelColumnsHierarchy
from sema.intake.profile import ProfileTools
from sema.intake.profile_utils import ColumnProfile
from sema.intake.shapes.base import RelationProfile, SlotBinding
from sema.intake.shapes.base_utils import failure, section_of, slot_relations, verdict
from sema.intake.verify.base import Failure, Verdict

ID = "level_columns"
SECTION = section_of(LevelColumnsHierarchy)
RELATION, LEVEL = "relation", "level_"
MIN_LEVELS = 2


def candidates(group: Sequence[RelationProfile], tools: ProfileTools) -> list[SlotBinding]:
    found = []
    for profile in group:
        levels = ordered_levels(profile)
        if len(levels) >= MIN_LEVELS and not _failures(profile, levels, tools):
            found.append(_binding(profile, levels))
    return found


def verify(
    binding: SlotBinding, group: Sequence[RelationProfile], tools: ProfileTools
) -> Verdict:
    found, failures = slot_relations(group, binding, (RELATION,))
    for profile in found:
        names = [binding.slots[slot] for slot in level_slots(binding)]
        missing = [name for name in names if profile.column(name) is None]
        if len(names) < MIN_LEVELS or missing:
            failures.append(failure("levels_exist", profile.ref.name, levels=len(names)))
        else:
            failures += _failures(profile, names, tools)
    return verdict(failures, {})


def ordered_levels(profile: RelationProfile) -> list[str]:
    filled = [c for c in profile.columns if c.distinct_count > 0 and c.null_rate < 1.0]
    return [c.name for c in sorted(filled, key=_depth_key)]


def level_slots(binding: SlotBinding) -> list[str]:
    depths = sorted(
        int(slot.removeprefix(LEVEL))
        for slot in binding.slots
        if slot.startswith(LEVEL) and slot.removeprefix(LEVEL).isdigit()
    )
    return [f"{LEVEL}{depth}" for depth in depths]


def _depth_key(column: ColumnProfile) -> tuple[float, int]:
    return column.null_rate, column.distinct_count


def _is_interior(column: ColumnProfile | None) -> bool:
    return column is not None and not column.is_constant and 0.0 < column.distinct_ratio < 1.0


def _failures(
    profile: RelationProfile, levels: Sequence[str], tools: ProfileTools
) -> list[Failure]:
    relation = profile.ref.name
    if not any(_is_interior(profile.column(name)) for name in levels):
        return [failure("interior_level", relation, levels=len(levels))]
    checked = tools.check_levels(profile.ref, levels)
    failures = []
    if checked.misplaced_nulls:
        failures.append(failure("nulls_below_only", relation, rows=checked.misplaced_nulls))
    if checked.multi_parent:
        failures.append(failure("one_path_per_node", relation, nodes=checked.multi_parent))
    return failures


def _binding(profile: RelationProfile, levels: Sequence[str]) -> SlotBinding:
    ref_name = profile.ref.name
    slots = {RELATION: ref_name} | {f"{LEVEL}{i}": name for i, name in enumerate(levels, 1)}
    return SlotBinding(template=ID, slots=slots, relations=(ref_name,))
