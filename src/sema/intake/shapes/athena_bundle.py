"""athena_bundle: eligible exactly when every required Athena table is landed.

The Athena loader fixes the table names, so this template may test them.
"""

from __future__ import annotations

from collections.abc import Sequence

from sema.binding.spec_sections import ConceptsSpec, ValueSetsSpec
from sema.intake.loaders.athena import REQUIRED_TABLES
from sema.intake.profile import ProfileTools
from sema.intake.shapes.base import RelationProfile, Slot, SlotBinding
from sema.intake.shapes.base_utils import (
    fixed_binding,
    fixed_failures,
    fixed_relation_sets,
    section_of,
    verdict,
)
from sema.intake.verify.base import FloorValue, Verdict, classed

ID = "athena_bundle"
SLOTS = tuple(Slot(name, "relation") for name in REQUIRED_TABLES)
# The importer scopes relationships to concepts, so the term-level relationship table, like the
# ancestor closure no hierarchy form expresses, stays staged directly by the loader (importer gap
# bug-757, S4). No term details: the concept table holds no definition or external reference;
# its validity columns are lifecycle roles of its values.
SECTIONS = (section_of(ConceptsSpec), section_of(ValueSetsSpec))
FLOORS: dict[str, FloorValue] = {}
GATES: frozenset[str] = frozenset()
FINDINGS: frozenset[str] = frozenset()
CLASSES = classed(GATES, FINDINGS)


def candidates(group: Sequence[RelationProfile], tools: ProfileTools) -> list[SlotBinding]:
    sets = fixed_relation_sets(group, REQUIRED_TABLES)
    return [fixed_binding(ID, relations, group) for relations in sets]


def verify(
    binding: SlotBinding, group: Sequence[RelationProfile], tools: ProfileTools
) -> Verdict:
    return verdict(fixed_failures(group, binding, REQUIRED_TABLES), FLOORS, CLASSES)
