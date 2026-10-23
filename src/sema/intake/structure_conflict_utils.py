"""Bound entries that fill one single-valued spec section reopen at Structure (decision 18).

Assembly takes one binding per section, so two bindings filling the same section would
leave a run with no open entry and no spec. Every relation of every such binding moves from
bound to open instead, offering its own binding, so a reviewer keeps one and ignores the rest.
"""

from __future__ import annotations

from sema.intake.eligibility_spec_utils import JOINT_TEMPLATES, MERGED_SECTIONS
from sema.intake.shapes import filled_sections
from sema.intake.shapes.precedence import STAGE, member
from sema.intake.structure import BoundEntry, StructureResult
from sema.intake.structure_hint_utils import SECTION_SEPARATOR
from sema.intake.verify.base import Ambiguity


def _sections(entry: BoundEntry) -> set[str]:
    heads = {name.partition(SECTION_SEPARATOR)[0] for name in filled_sections(entry.binding)}
    return heads - MERGED_SECTIONS


def _conflicts(bound: tuple[BoundEntry, ...]) -> dict[str, list[int]]:
    """Every single-valued section two or more entries fill, with their indices."""
    filling: dict[str, list[int]] = {}
    for index, entry in enumerate(bound):
        if entry.binding.template in JOINT_TEMPLATES:
            continue
        for section in sorted(_sections(entry)):
            filling.setdefault(section, []).append(index)
    return {section: indices for section, indices in filling.items() if len(indices) > 1}


def _reasons(conflicts: dict[str, list[int]]) -> dict[int, str]:
    """Each conflicting entry's reason, naming every section it competes for."""
    found: dict[int, list[str]] = {}
    for section, indices in sorted(conflicts.items()):
        for index in indices:
            found.setdefault(index, []).append(f"{len(indices)} bindings fill section {section}")
    return {index: "; ".join(parts) for index, parts in found.items()}


def open_section_conflicts(structure: StructureResult) -> StructureResult:
    """Assembly takes one binding per section, so competing bindings go back to Review."""
    reasons = _reasons(_conflicts(structure.bound))
    if not reasons:
        return structure
    reopened = [
        Ambiguity(
            stage=STAGE,
            subject=relation,
            members=(member(structure.bound[index].binding),),
            reason=reason,
        )
        for index, reason in sorted(reasons.items())
        for relation in structure.bound[index].binding.relations
    ]
    kept = tuple(e for i, e in enumerate(structure.bound) if i not in reasons)
    update = {"bound": kept, "ambiguities": (*structure.ambiguities, *reopened)}
    return structure.model_copy(update=update)
