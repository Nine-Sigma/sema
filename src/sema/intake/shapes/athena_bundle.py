"""athena_bundle: eligible exactly when every required Athena table is landed.

The Athena loader fixes the table names, so this template may test them.
"""

from __future__ import annotations

from collections.abc import Sequence

from sema.intake.loaders.athena import REQUIRED_TABLES
from sema.intake.profile import ProfileTools
from sema.intake.shapes.base import RelationProfile, SlotBinding
from sema.intake.shapes.base_utils import (
    fixed_binding,
    fixed_failures,
    fixed_relation_sets,
    verdict,
)
from sema.intake.verify.base import Verdict

ID = "athena_bundle"


def candidates(group: Sequence[RelationProfile], tools: ProfileTools) -> list[SlotBinding]:
    sets = fixed_relation_sets(group, REQUIRED_TABLES)
    return [fixed_binding(ID, relations) for relations in sets]


def verify(
    binding: SlotBinding, group: Sequence[RelationProfile], tools: ProfileTools
) -> Verdict:
    return verdict(fixed_failures(group, binding, REQUIRED_TABLES), {})
