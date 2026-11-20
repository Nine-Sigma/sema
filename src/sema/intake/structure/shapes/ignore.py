"""ignore: relations with no structural content, decided from profiles before any template.

Reasons, in precedence order: ``no_header`` (the loader landed no header row), ``no_rows``,
and ``boilerplate`` (repeated unchanged in at least the share floor of the other files). A
reviewer's ``ignore`` decision records ``human``, which no profile re-derives. The binding
records which templates would otherwise have matched, so a reviewer can rebind it. The share
floor and the identity test are gates (S5 decision 22); a boilerplate binding states its
share as the finding ``boilerplate_share``.
"""

from __future__ import annotations

from collections.abc import Sequence

from sema.intake.land.profile import ProfileTools
from sema.intake.structure.shapes.base import RelationProfile, SlotBinding
from sema.intake.structure.shapes.base_utils import IN_GROUP, failure, find, finding, verdict
from sema.intake.structure.shapes.ignore_utils import FULL_RESOLVE, other_multi_groups, repeated_share
from sema.intake.verify.base import Finding, FloorValue, Verdict, classed

ID = "ignore"
REASON, WOULD_MATCH = "reason", "would_match"
NO_HEADER, NO_ROWS, BOILERPLATE, HUMAN = "no_header", "no_rows", "boilerplate", "human"
REASONS = (NO_HEADER, NO_ROWS, BOILERPLATE, HUMAN)
# Measured: the ACRIS options sheet is identical in 15 of 15 workbooks (share 1.0 of the
# other workbooks); an ACRIS column dictionary sheet matches at most its one twin of 14 (0.07).
BOILERPLATE_GROUP_SHARE_FLOOR = 0.8
FLOORS: dict[str, FloorValue] = {
    "BOILERPLATE_GROUP_SHARE_FLOOR": BOILERPLATE_GROUP_SHARE_FLOOR,
    "FULL_RESOLVE": FULL_RESOLVE,
}
GATES = frozenset({"BOILERPLATE_GROUP_SHARE_FLOOR", "FULL_RESOLVE"})
FINDINGS: frozenset[str] = frozenset()
CLASSES = classed(GATES, FINDINGS)
BOILERPLATE_SHARE = "boilerplate_share"
MATCH_SEPARATOR = ","


def candidates(group: Sequence[RelationProfile], tools: ProfileTools) -> list[SlotBinding]:
    found = []
    for profile in group:
        reading = _reading(profile, group, tools)
        if reading is not None:
            reason, findings = reading
            found.append(_binding(profile, reason, would_match(profile, tools), findings))
    return found


def verify(
    binding: SlotBinding, group: Sequence[RelationProfile], tools: ProfileTools
) -> Verdict:
    ref_name = binding.relations[0] if len(binding.relations) == 1 else None
    profile = find(group, ref_name)
    if profile is None:
        return verdict([failure(IN_GROUP, ref_name or "", slot=REASON)], FLOORS, CLASSES)
    claimed, reading = binding.slots.get(REASON), _reading(profile, group, tools)
    reason, findings = reading if reading is not None else (None, [])
    if claimed != reason:
        failed = failure(REASON, profile.ref.name, claimed=claimed, found=reason or "")
        return verdict([failed], FLOORS, CLASSES, findings)
    return verdict([], FLOORS, CLASSES, findings)


def reason_for(
    profile: RelationProfile, group: Sequence[RelationProfile], tools: ProfileTools
) -> str | None:
    reading = _reading(profile, group, tools)
    return None if reading is None else reading[0]


def _reading(
    profile: RelationProfile, group: Sequence[RelationProfile], tools: ProfileTools
) -> tuple[str, list[Finding]] | None:
    if profile.headerless:
        return NO_HEADER, []
    if profile.row_count == 0:
        return NO_ROWS, []
    others = other_multi_groups(profile, group) or []
    share = repeated_share(profile, others, BOILERPLATE_GROUP_SHARE_FLOOR, tools)
    if share is None:
        return None
    return BOILERPLATE, [finding(BOILERPLATE_SHARE, profile.ref.name, None, share)]


def would_match(profile: RelationProfile, tools: ProfileTools) -> list[str]:
    # Imported here because the package's template tuple includes this module.
    from sema.intake.structure.shapes import TEMPLATES

    return [
        template.ID
        for template in TEMPLATES
        if template.ID != ID and template.candidates([profile], tools)
    ]


def decided(profile: RelationProfile, tools: ProfileTools) -> SlotBinding:
    return _binding(profile, HUMAN, would_match(profile, tools))


def _binding(
    profile: RelationProfile,
    reason: str,
    matches: Sequence[str],
    findings: Sequence[Finding] = (),
) -> SlotBinding:
    ref_name = profile.ref.name
    slots = {REASON: reason, WOULD_MATCH: MATCH_SEPARATOR.join(matches)}
    return SlotBinding(template=ID, slots=slots, relations=(ref_name,), findings=tuple(findings))
