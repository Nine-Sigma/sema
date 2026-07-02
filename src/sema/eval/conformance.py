"""F1: deterministic contract-conformance gate (bug-369).

The Slice-0 deterministic path is validated WITHOUT gold labels by re-verifying
every RESOLVED concept_id against the authoritative target vocabulary: it must be
present, valid, standard (when the binding requires it), and in the target
domain. This is the self-consistency assertion vs the target's own crosswalk —
it catches a resolver/store regression that emits a wrong-domain or non-standard
concept, which Gate D-lite's row/null checks cannot.

Lives under ``eval/`` (R29-allowlisted) so it may read the resolved
value-mapping rows. It stays vocabulary-agnostic: the standard flag and target
domain are read from the :class:`~sema.resolve.policy.ResolverPolicy`, never
hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from sema.resolve.policy import ResolverPolicy
from sema.resolve.value_mapping_store_utils import ResolutionStatus, ValueMapping
from sema.resolve.vocab_store_utils import ConceptRow


class ConceptLookup(Protocol):
    """The one vocab-store capability conformance needs (see ``concepts_by_ids``)."""

    def concepts_by_ids(self, ids: list[str]) -> dict[str, ConceptRow | None]: ...


@dataclass(frozen=True)
class ConformanceViolation:
    """One resolved decision whose concept breaks the target contract."""

    concept_id: int | None
    source_value: str
    reason: str


@dataclass(frozen=True)
class ConformanceReport:
    """Whether every resolved concept meets the target contract."""

    passed: bool
    violations: tuple[ConformanceViolation, ...]
    resolved_count: int
    no_map_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "resolved": self.resolved_count,
            "no_map": self.no_map_count,
            "violations": [
                {
                    "concept_id": v.concept_id,
                    "source_value": v.source_value,
                    "reason": v.reason,
                }
                for v in self.violations
            ],
        }


def assert_contract_conformance(
    mappings: Sequence[ValueMapping],
    lookup: ConceptLookup,
    policy: ResolverPolicy,
) -> ConformanceReport:
    """Re-verify every RESOLVED concept against the target vocab contract.

    ``mappings`` should be the CURRENT run's decisions (scoped to this
    run/property/policy/vocab release), not the whole historical store.
    """
    resolved = [m for m in mappings if m.resolution_status is ResolutionStatus.RESOLVED]
    no_map_count = sum(
        1 for m in mappings if m.resolution_status is ResolutionStatus.NO_MAP
    )
    ids = [str(m.concept_id) for m in resolved if m.concept_id is not None]
    concepts = lookup.concepts_by_ids(ids)
    violations = [
        ConformanceViolation(m.concept_id, m.normalized_source_value, reason)
        for m in resolved
        if (reason := _violation_reason(_lookup_row(m.concept_id, concepts), policy))
    ]
    return ConformanceReport(
        passed=not violations,
        violations=tuple(violations),
        resolved_count=len(resolved),
        no_map_count=no_map_count,
    )


def _lookup_row(
    concept_id: int | None, concepts: dict[str, ConceptRow | None]
) -> ConceptRow | None:
    return None if concept_id is None else concepts.get(str(concept_id))


def _violation_reason(row: ConceptRow | None, policy: ResolverPolicy) -> str | None:
    if row is None:
        return "resolved concept_id absent from target vocabulary"
    if policy.require_standard and row.standard != policy.standard_flag:
        return f"resolved concept is not standard (standard={row.standard!r})"
    if row.invalid_reason is not None:
        return f"resolved concept is invalid (invalid_reason={row.invalid_reason!r})"
    if policy.target_domain is not None and row.domain != policy.target_domain:
        return (
            f"resolved concept domain {row.domain!r} != "
            f"target domain {policy.target_domain!r}"
        )
    return None
