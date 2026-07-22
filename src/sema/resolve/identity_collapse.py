"""S1-10 — Stage B deterministic identity collapse (generic, R29-scanned).

Reads the identity registry, computes the deterministic collapse plan
(:mod:`identity_collapse_utils`), and applies it as an ``entity_id`` remap. This
is the *identity* half of Stage B; the persisted child rows are brought into line
by rebuilding them through the collapsed registry (the FK-closed compiler already
retires orphaned parents via ``SELECT DISTINCT entity_id`` and recomputes each
child FK from ``(namespace, key) -> entity_id`` — see the H1 semantics in the
Slice-1 plan). The rebuild is orchestrated in the pipeline layer, not here.

Domain-generic (D6/R29): the "which source namespaces share an identity
namespace" grouping is supplied by the caller as data; nothing here names an
institution, study, or OMOP literal.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from sema.resolve.identity_collapse_utils import CollapsePlan, compute_collapse_plan
from sema.resolve.identity_registry import IdentityRegistry


@dataclass(frozen=True)
class CollapseResult:
    """The outcome of one Stage B collapse pass over the registry."""

    plan: CollapsePlan
    remapped_row_count: int

    @property
    def collapsed_person_count(self) -> int:
        return self.plan.collapsed_person_count

    @property
    def retired_entity_ids(self) -> tuple[int, ...]:
        return self.plan.retired_entity_ids


def collapse_identities(
    registry: IdentityRegistry, *, namespace_grouping: Mapping[str, str]
) -> CollapseResult:
    """Collapse same-identity-namespace shared keys onto one canonical id.

    Idempotent: an already-collapsed registry yields an empty plan and zero
    remaps. Never collapses across identity namespaces (the over-collapse trap),
    because the plan buckets by ``(identity_namespace, key)``.
    """
    plan = compute_collapse_plan(
        registry.read_all(), namespace_grouping=namespace_grouping
    )
    remapped = registry.remap_entity_ids(plan.remaps)
    return CollapseResult(plan=plan, remapped_row_count=remapped)
