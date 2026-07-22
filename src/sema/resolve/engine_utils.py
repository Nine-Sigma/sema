"""Per-code resolution result, context, and the §4 EMIT builders (US-006).

All field-level construction that names the resolved concept lives here, *not*
in :mod:`sema.resolve.engine` — the engine is an R29-scanned core path that must
stay free of domain literals, so the orchestrator delegates every concept-level
detail to this (unscanned) helper module.
"""

from __future__ import annotations

from dataclasses import dataclass

from sema.compile.compiler_utils import StagingDecision
from sema.models.planner.lifecycle import Status
from sema.models.planner.mapping_plan import MappingAssertion
from sema.models.planner.patterns import MappingPattern, VocabLookup
from sema.models.planner.provenance import Provenance
from sema.resolve.disambiguate import Disambiguator
from sema.resolve.policy import ResolverPolicy
from sema.resolve.value_mapping_store_utils import ResolutionStatus, ValueMapping
from sema.resolve.vocab_store_utils import ConceptRow

# Deterministic SQL-walk outcomes are fully confident; an escalated >1-survivor
# tie is not (it carries review_pending status → Zone-2, "reported, not gated").
AUTO_CONFIDENCE = 1.0
NO_MAP_CONFIDENCE = 1.0
AMBIGUOUS_CONFIDENCE = 0.5

@dataclass(frozen=True)
class ResolveTrace:
    """Stage counts from one resolve pass, used to explain a NO_MAP outcome.

    ``n_candidates`` is the source-vocabulary match count (step 1a);
    ``n_standardized`` is the deduped standard/valid target count before the
    domain gate (step 2). Zero survivors + these two counts pinpoint WHERE the
    walk ended without hardcoding any domain literal here.
    """

    n_candidates: int
    n_standardized: int


@dataclass(frozen=True)
class CodeResolution:
    """One distinct source code's deterministic decision."""

    source_code: str
    resolution_status: ResolutionStatus
    concept_id: int | None
    status: Status
    confidence: float
    no_map_reason: str | None
    survivors: tuple[ConceptRow, ...]
    chosen: ConceptRow | None


@dataclass(frozen=True)
class ResolveContext:
    """Per-job refs and provenance threaded into store rows and assertions.

    These are data supplied by the caller (the US-009 producer / a manifest),
    never literals owned by the resolver engine.
    """

    source_field_ref: str
    source_value_ref: str
    target_property_ref: str
    target_field: str
    domain_constraint_ref: str
    vocabulary_ref: str
    vocab_binding: str
    vocab_release: str
    resolver_policy_ref: str
    run_id: str
    provenance: Provenance
    effective_date_ref: str | None = None


def dedupe_concepts(concepts: list[ConceptRow]) -> list[ConceptRow]:
    """Drop duplicate concept rows by identity, preserving first-seen order."""
    seen: set[str] = set()
    unique: list[ConceptRow] = []
    for concept in concepts:
        if concept.id in seen:
            continue
        seen.add(concept.id)
        unique.append(concept)
    return unique


def classify_resolution(
    source_code: str,
    survivors: list[ConceptRow],
    disambiguate: Disambiguator,
    trace: ResolveTrace,
    policy: ResolverPolicy,
) -> CodeResolution:
    """Map surviving standard candidates to a §1.5(c) Zone decision."""
    if not survivors:
        return _no_map(source_code, trace, policy)
    if len(survivors) == 1:
        return _resolved(source_code, survivors[0], Status.auto_accepted, AUTO_CONFIDENCE)
    chosen = disambiguate(survivors) or min(survivors, key=lambda c: int(c.id))
    return _resolved(source_code, chosen, Status.review_pending, AMBIGUOUS_CONFIDENCE)


def _resolved(
    source_code: str, chosen: ConceptRow, status: Status, confidence: float
) -> CodeResolution:
    return CodeResolution(
        source_code=source_code,
        resolution_status=ResolutionStatus.RESOLVED,
        concept_id=int(chosen.id),
        status=status,
        confidence=confidence,
        no_map_reason=None,
        survivors=(chosen,),
        chosen=chosen,
    )


def _no_map(
    source_code: str, trace: ResolveTrace, policy: ResolverPolicy
) -> CodeResolution:
    return CodeResolution(
        source_code=source_code,
        resolution_status=ResolutionStatus.NO_MAP,
        concept_id=None,
        status=Status.auto_accepted,
        confidence=NO_MAP_CONFIDENCE,
        no_map_reason=_no_map_reason(trace, policy),
        survivors=(),
        chosen=None,
    )


def _no_map_reason(trace: ResolveTrace, policy: ResolverPolicy) -> str:
    """Explain WHERE the deterministic walk ended, using policy data only.

    Reason strings interpolate the policy's source vocabulary, standardizing
    relationship, and target domain — never a hardcoded domain literal — so this
    stays vocabulary-agnostic (and R29-clean).
    """
    if trace.n_candidates == 0:
        return f"source code not found in vocabulary '{policy.source_vocabulary}'"
    if trace.n_standardized == 0:
        return (
            f"no standard, valid '{policy.maps_to_relationship}' target "
            "(no curated crosswalk)"
        )
    return (
        "standardized target(s) fell outside target domain "
        f"'{policy.target_domain}'"
    )


def build_value_mapping(
    resolution: CodeResolution, policy: ResolverPolicy, context: ResolveContext
) -> ValueMapping:
    """Project a resolution onto a §1.5(a) value-mapping-store row."""
    return ValueMapping(
        source_vocabulary=policy.source_vocabulary,
        normalized_source_value=resolution.source_code,
        target_property_ref=context.target_property_ref,
        target_field=context.target_field,
        vocab_binding=context.vocab_binding,
        concept_id=resolution.concept_id,
        vocab_release=context.vocab_release,
        valid_start=None,
        valid_end=None,
        resolution_status=resolution.resolution_status,
        no_map_reason=resolution.no_map_reason,
        confidence=resolution.confidence,
        status=resolution.status,
        resolver_policy_ref=context.resolver_policy_ref,
        run_id=context.run_id,
    )


def build_vocab_lookup_assertion(
    resolution: CodeResolution, policy: ResolverPolicy, context: ResolveContext
) -> MappingAssertion:
    """Emit the §4 step-5 VOCAB_LOOKUP assertion for a resolved survivor."""
    if resolution.resolution_status is not ResolutionStatus.RESOLVED:
        raise ValueError("VOCAB_LOOKUP assertion requires a RESOLVED resolution")
    payload = VocabLookup(
        vocabulary_ref=context.vocabulary_ref,
        source_value_ref=context.source_value_ref,
        domain_constraint_ref=context.domain_constraint_ref,
        require_standard=policy.require_standard,
        allow_zero_default=policy.allow_zero_default,
        resolver_policy_ref=context.resolver_policy_ref,
        effective_date_ref=context.effective_date_ref,
    )
    return MappingAssertion(
        id=_assertion_id(resolution, policy, context),
        source_field_ref=context.source_field_ref,
        target_property_ref=context.target_property_ref,
        pattern=MappingPattern.VOCAB_LOOKUP,
        payload=payload,
        confidence=resolution.confidence,
        provenance=context.provenance,
        status=resolution.status,
    )


def _assertion_id(
    resolution: CodeResolution, policy: ResolverPolicy, context: ResolveContext
) -> str:
    return (
        f"vocab_lookup:{policy.source_vocabulary}:"
        f"{resolution.source_code}->{context.target_field}"
    )


def staging_decision_from_value_mapping(mapping: ValueMapping) -> StagingDecision:
    """Project a §1.5(a) store row onto a generic compiler input (US-010).

    Lives here (an unscanned sibling) because it reads the store's concept-id
    field; the R29-scanned compiler only ever sees the neutral
    :class:`StagingDecision`.
    """
    return StagingDecision(
        normalized_source_value=mapping.normalized_source_value,
        target_value=mapping.concept_id,
        resolution_status=mapping.resolution_status.value,
        no_map_reason=mapping.no_map_reason,
        status=_status_value(mapping.status),
    )


def _status_value(status: Status) -> str:
    return status.value if isinstance(status, Status) else str(status)
