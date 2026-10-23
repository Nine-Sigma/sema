"""Bound entries that fill one single-valued spec section reopen at Structure (decision 18).

Assembly takes one binding per section, so two bindings filling the same section would
leave a run with no open entry and no spec. Every relation of every such binding moves from
bound to open instead, so a reviewer keeps one and ignores the rest. A relation an ask settled
offers every member that ask offered, the settled choice first (decision 21), so a reviewer
can choose a member the model rejected; one bound by code offers its own binding.
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


def _reopened(relation: str, bound: str, settled: Ambiguity | None, reason: str) -> Ambiguity:
    """The settled entry, open again with its members ranked from the binding it settled on."""
    if settled is None or bound not in settled.members:
        return Ambiguity(stage=STAGE, subject=relation, members=(bound,), reason=reason)
    others = tuple(m for m in settled.members if m != bound)
    return settled.model_copy(update=dict(
        members=(bound, *others), choice=None, confidence=None, source=None, call_key=None,
        reason=reason,
    ))


def open_section_conflicts(structure: StructureResult) -> StructureResult:
    """Assembly takes one binding per section, so competing bindings go back to Review."""
    reasons = _reasons(_conflicts(structure.bound))
    if not reasons:
        return structure
    settled = {entry.subject: entry for entry in structure.settled if entry.stage == STAGE}
    reopened = [
        _reopened(relation, member(entry.binding), settled.get(relation), reason)
        for index, reason in sorted(reasons.items())
        for entry in (structure.bound[index],)
        for relation in entry.binding.relations
    ]
    moved = {entry.subject for entry in reopened}
    update = {
        "bound": tuple(e for i, e in enumerate(structure.bound) if i not in reasons),
        "ambiguities": (*structure.ambiguities, *reopened),
        "settled": tuple(e for e in structure.settled if e.subject not in moved),
    }
    return structure.model_copy(update=update)
