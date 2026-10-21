"""ignore: relations with no structural content, decided from profiles before any template.

Reasons, in precedence order: ``no_header`` (the loader landed no header row), ``no_rows``,
and ``boilerplate`` (repeated unchanged in at least the share floor of the other files). The
binding records which templates would otherwise have matched, so a reviewer can rebind it.
"""

from __future__ import annotations

from collections.abc import Sequence

from sema.intake.profile import ProfileTools
from sema.intake.shapes.base import RelationProfile, SlotBinding
from sema.intake.shapes.base_utils import IN_GROUP, failure, find, verdict
from sema.intake.shapes.ignore_utils import other_multi_groups, repeated_in_share
from sema.intake.verify.base import FloorValue, Verdict

ID = "ignore"
REASON, WOULD_MATCH = "reason", "would_match"
NO_HEADER, NO_ROWS, BOILERPLATE = "no_header", "no_rows", "boilerplate"
REASONS = (NO_HEADER, NO_ROWS, BOILERPLATE)
# Measured: the ACRIS options sheet is identical in 15 of 15 workbooks (share 1.0 of the
# other workbooks); an ACRIS column dictionary sheet matches at most its one twin of 14 (0.07).
BOILERPLATE_GROUP_SHARE_FLOOR = 0.8
FLOORS: dict[str, FloorValue] = {"BOILERPLATE_GROUP_SHARE_FLOOR": BOILERPLATE_GROUP_SHARE_FLOOR}
MATCH_SEPARATOR = ","


def candidates(group: Sequence[RelationProfile], tools: ProfileTools) -> list[SlotBinding]:
    found = []
    for profile in group:
        reason = reason_for(profile, group, tools)
        if reason is not None:
            found.append(_binding(profile, reason, would_match(profile, tools)))
    return found


def verify(
    binding: SlotBinding, group: Sequence[RelationProfile], tools: ProfileTools
) -> Verdict:
    ref_name = binding.relations[0] if len(binding.relations) == 1 else None
    profile = find(group, ref_name)
    if profile is None:
        return verdict([failure(IN_GROUP, ref_name or "", slot=REASON)], FLOORS)
    claimed, reason = binding.slots.get(REASON), reason_for(profile, group, tools)
    if claimed != reason:
        found = reason or ""
        return verdict([failure(REASON, profile.ref.name, claimed=claimed, found=found)], FLOORS)
    return verdict([], FLOORS)


def reason_for(
    profile: RelationProfile, group: Sequence[RelationProfile], tools: ProfileTools
) -> str | None:
    if profile.headerless:
        return NO_HEADER
    if profile.row_count == 0:
        return NO_ROWS
    others = other_multi_groups(profile, group)
    if others and repeated_in_share(profile, others, BOILERPLATE_GROUP_SHARE_FLOOR, tools):
        return BOILERPLATE
    return None


def would_match(profile: RelationProfile, tools: ProfileTools) -> list[str]:
    # Imported here because the package's template tuple includes this module.
    from sema.intake.shapes import TEMPLATES

    return [
        template.ID
        for template in TEMPLATES
        if template.ID != ID and template.candidates([profile], tools)
    ]


def _binding(profile: RelationProfile, reason: str, matches: Sequence[str]) -> SlotBinding:
    ref_name = profile.ref.name
    slots = {REASON: reason, WOULD_MATCH: MATCH_SEPARATOR.join(matches)}
    return SlotBinding(template=ID, slots=slots, relations=(ref_name,))
