"""semsql: eligible exactly when the semsql loader's fixed relations are landed.

This and athena_bundle are the only templates that test relation names, because their
loaders fix the names.
"""

from __future__ import annotations

from collections.abc import Sequence

from sema.intake.loaders.semsql import REQUIRED_BASE_TABLES
from sema.intake.loaders.semsql_utils import DERIVED_RELATIONS, TERM_LABEL_COLUMN, TERMS_RELATION
from sema.intake.profile import ProfileTools
from sema.intake.shapes.base import RelationProfile, SlotBinding
from sema.intake.shapes.base_utils import (
    failure,
    find,
    fixed_binding,
    fixed_failures,
    fixed_relation_sets,
    verdict,
)
from sema.intake.verify.base import Failure, FloorValue, Verdict

ID = "semsql"
REQUIRED = (*DERIVED_RELATIONS, *REQUIRED_BASE_TABLES)
# Measured: A3 fixture semsql_terms labels 50 of 50 classes (coverage 1.0).
LABEL_COVERAGE_FLOOR = 0.9
FLOORS: dict[str, FloorValue] = {"LABEL_COVERAGE_FLOOR": LABEL_COVERAGE_FLOOR}


def candidates(group: Sequence[RelationProfile], tools: ProfileTools) -> list[SlotBinding]:
    return [
        fixed_binding(ID, relations, group)
        for relations in fixed_relation_sets(group, REQUIRED)
        if not _coverage_failures(relations[TERMS_RELATION])
    ]


def verify(
    binding: SlotBinding, group: Sequence[RelationProfile], tools: ProfileTools
) -> Verdict:
    failures = fixed_failures(group, binding, REQUIRED)
    terms = find(group, binding.slots.get(TERMS_RELATION))
    if not failures and terms is not None:
        failures = _coverage_failures(terms)
    return verdict(failures, FLOORS)


def _coverage_failures(terms: RelationProfile) -> list[Failure]:
    label = terms.column(TERM_LABEL_COLUMN)
    coverage = 1.0 - label.null_rate if label is not None and terms.row_count else 0.0
    if coverage < LABEL_COVERAGE_FLOOR:
        return [failure("label_coverage", terms.ref.name, TERM_LABEL_COLUMN, coverage=coverage)]
    return []
