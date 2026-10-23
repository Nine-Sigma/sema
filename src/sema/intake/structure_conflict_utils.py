"""Bound entries that fill one single-valued spec section reopen at Structure (decision 18).

Assembly takes one binding per section, so two bindings filling the same section would
leave a run with no open entry and no spec. Every relation of every such binding moves from
bound to open instead, offering its own binding, so a reviewer keeps one and ignores the rest.
"""

from __future__ import annotations

from collections import Counter

from sema.intake.eligibility_spec_utils import JOINT_TEMPLATES, MERGED_SECTIONS
from sema.intake.shapes import filled_sections
from sema.intake.shapes.precedence import STAGE, member
from sema.intake.structure import BoundEntry, StructureResult
from sema.intake.structure_hint_utils import SECTION_SEPARATOR
from sema.intake.verify.base import Ambiguity


def _sections(entry: BoundEntry) -> set[str]:
    heads = {name.partition(SECTION_SEPARATOR)[0] for name in filled_sections(entry.binding)}
    return heads - MERGED_SECTIONS


def _conflicts(bound: tuple[BoundEntry, ...]) -> dict[int, str]:
    """The index of every entry sharing a single-valued section, with that section."""
    filling: dict[str, list[int]] = {}
    for index, entry in enumerate(bound):
        if entry.binding.template in JOINT_TEMPLATES:
            continue
        for section in _sections(entry):
            filling.setdefault(section, []).append(index)
    return {
        index: section
        for section, indices in sorted(filling.items())
        if len(indices) > 1
        for index in indices
    }


def open_section_conflicts(structure: StructureResult) -> StructureResult:
    conflicts = _conflicts(structure.bound)
    if not conflicts:
        return structure
    counts = Counter(conflicts.values())
    reopened = [
        Ambiguity(
            stage=STAGE,
            subject=relation,
            members=(member(structure.bound[index].binding),),
            reason=f"{counts[section]} bindings fill section {section}",
        )
        for index, section in sorted(conflicts.items())
        for relation in structure.bound[index].binding.relations
    ]
    kept = tuple(e for i, e in enumerate(structure.bound) if i not in conflicts)
    update = {"bound": kept, "ambiguities": (*structure.ambiguities, *reopened)}
    return structure.model_copy(update=update)
