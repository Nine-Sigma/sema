"""US-009 helpers: project value-mapping decisions into a column mapping.

The VOCAB_LOOKUP pattern maps a whole source column to a target property; the
per-code ``source_value -> concept_id`` decisions live in the value-mapping
store (written solely by US-006) and are inlined later by the compiler (US-010).
This module reads those decisions and folds them into ONE column-level
``MappingAssertion`` plus its ``FieldMap`` projection.

Every concept-level detail (the VocabLookup payload) is reused from
:mod:`sema.resolve.engine_utils`, so this module names no domain literal.
"""

from __future__ import annotations

from sema.models.planner.field_map import FieldMap
from sema.models.planner.lifecycle import Status
from sema.models.planner.mapping_plan import MappingAssertion
from sema.resolve.engine_utils import (
    CodeResolution,
    ResolveContext,
    build_vocab_lookup_assertion,
)
from sema.resolve.policy import ResolverPolicy
from sema.resolve.value_mapping_store_utils import ResolutionStatus, ValueMapping


class NoResolvedDecisionError(ValueError):
    """Raised when no RESOLVED decision exists for the binding being produced."""


def decisions_for_binding(
    decisions: list[ValueMapping], policy: ResolverPolicy, context: ResolveContext
) -> list[ValueMapping]:
    """Keep only decisions belonging to this source-vocabulary / target binding."""
    return [
        d
        for d in decisions
        if d.source_vocabulary == policy.source_vocabulary
        and d.target_property_ref == context.target_property_ref
        and d.resolver_policy_ref == context.resolver_policy_ref
        and d.vocab_release == context.vocab_release
    ]


def build_column_assertion(
    decisions: list[ValueMapping], policy: ResolverPolicy, context: ResolveContext
) -> MappingAssertion:
    """Fold the per-code decisions into one column-level VOCAB_LOOKUP assertion."""
    resolved = [
        d for d in decisions if d.resolution_status is ResolutionStatus.RESOLVED
    ]
    if not resolved:
        raise NoResolvedDecisionError(
            f"no RESOLVED decision for {context.target_property_ref!r}"
        )
    column = CodeResolution(
        source_code=context.source_field_ref,
        resolution_status=ResolutionStatus.RESOLVED,
        concept_id=None,
        status=_aggregate_status(resolved),
        confidence=min(d.confidence for d in resolved),
        no_map_reason=None,
        survivors=(),
        chosen=None,
    )
    return build_vocab_lookup_assertion(column, policy, context)


def field_map_from_assertion(assertion: MappingAssertion) -> FieldMap:
    """Project the winning column assertion into its FieldMap (§1.5(d) status)."""
    return FieldMap(
        target_field_ref=assertion.target_property_ref,
        pattern=assertion.pattern,
        payload=assertion.payload,
        status=assertion.status,
    )


def _aggregate_status(resolved: list[ValueMapping]) -> Status:
    if any(d.status is Status.review_pending for d in resolved):
        return Status.review_pending
    return Status.auto_accepted
