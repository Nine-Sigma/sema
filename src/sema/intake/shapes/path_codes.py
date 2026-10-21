"""path_codes: a code column whose separator splits every code into non-empty segments.

Prefixes that are not codes themselves are implicit; the binding carries their count, never
the prefixes, because a prefix is a value.
"""

from __future__ import annotations

from collections.abc import Sequence

from sema.intake.profile import ProfileTools
from sema.intake.profile_utils import ColumnProfile
from sema.intake.shapes.base import RelationProfile, SlotBinding, is_id, is_text
from sema.intake.shapes.base_utils import failure, slot_columns, slot_relations, verdict
from sema.intake.verify.base import Failure, Verdict

ID = "path_codes"
RELATION, CODE, SEPARATOR = "relation", "code", "separator"
SEPARATORS = (".", "/", "-", ":", "|", ">")
Checked = tuple[list[Failure], dict[str, float | str]]


def candidates(group: Sequence[RelationProfile], tools: ProfileTools) -> list[SlotBinding]:
    found = []
    for profile in group:
        for code in (c for c in profile.columns if is_id(profile, c) and is_text(c)):
            for separator in SEPARATORS:
                failures, evidence = _check(profile, code, separator, tools)
                if not failures:
                    found.append(_binding(profile, code, separator, evidence))
    return found


def verify(
    binding: SlotBinding, group: Sequence[RelationProfile], tools: ProfileTools
) -> Verdict:
    found, failures = slot_relations(group, binding, (RELATION,))
    for profile in found:
        codes, missing = slot_columns(profile, binding, (CODE,))
        separator = binding.slots.get(SEPARATOR, "")
        failures += missing
        failures += [f for code in codes for f in _check(profile, code, separator, tools)[0]]
    return verdict(failures, {})


def _check(
    profile: RelationProfile, code: ColumnProfile, separator: str, tools: ProfileTools
) -> Checked:
    relation = profile.ref.name
    if not (is_id(profile, code) and is_text(code)):
        return [failure("code_is_id", relation, code.name)], {}
    if len(separator) != 1:
        return [failure("separator_is_one_character", relation, code.name)], {}
    checked = tools.check_paths(profile.ref, code.name, separator)
    failures = []
    if not checked.separated:
        failures.append(failure("separator_used", relation, code.name, separator=separator))
    if checked.malformed:
        failures.append(failure("segments_non_empty", relation, code.name, codes=checked.malformed))
    evidence: dict[str, float | str] = {
        "implicit_count": float(checked.implicit_count),
        "separated": float(checked.separated),
    }
    return failures, evidence


def _binding(
    profile: RelationProfile, code: ColumnProfile, separator: str, evidence: dict[str, float | str]
) -> SlotBinding:
    ref_name = profile.ref.name
    return SlotBinding(
        template=ID,
        slots={RELATION: ref_name, CODE: code.name, SEPARATOR: separator},
        evidence=evidence,
        relations=(ref_name,),
    )
