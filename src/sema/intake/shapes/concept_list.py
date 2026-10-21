"""concept_list: one relation with an id signature and a label signature.

Which columns are the id and the label are roles; the section is resolved at Roles.
"""

from __future__ import annotations

from collections.abc import Sequence

from sema.intake.profile import ProfileTools
from sema.intake.shapes.base import RelationProfile, SlotBinding, id_columns, label_columns
from sema.intake.shapes.base_utils import SIGNATURE_FLOORS, failure, slot_relations, verdict
from sema.intake.verify.base import Failure, Verdict

ID = "concept_list"
RELATION = "relation"


def candidates(group: Sequence[RelationProfile], tools: ProfileTools) -> list[SlotBinding]:
    return [_binding(profile) for profile in group if not _failures(profile)]


def verify(
    binding: SlotBinding, group: Sequence[RelationProfile], tools: ProfileTools
) -> Verdict:
    found, failures = slot_relations(group, binding, (RELATION,))
    for profile in found:
        failures += _failures(profile)
    return verdict(failures, SIGNATURE_FLOORS)


def _failures(profile: RelationProfile) -> list[Failure]:
    failures = []
    if not id_columns(profile):
        failures.append(failure("id_signature", profile.ref.name, rows=profile.row_count))
    if not label_columns(profile):
        failures.append(failure("label_signature", profile.ref.name))
    return failures


def _binding(profile: RelationProfile) -> SlotBinding:
    ref_name = profile.ref.name
    return SlotBinding(template=ID, slots={RELATION: ref_name}, relations=(ref_name,))
