"""Helpers for `load_target`: enrichment decisions, card hashes, aggregate version."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime, timezone

from sema.models.target.completeness import SemanticCompleteness
from sema.models.target.context_card import LoadedContextCard, TargetContextCard
from sema.models.target.descriptor import TargetModelDescriptor
from sema.models.target.entity import TargetEntityDecl
from sema.models.target.enrichment import (
    EnrichmentDecisionRecord,
    EnrichmentStatus,
    Facet,
    FacetDecision,
)
from sema.models.target.refs import VocabularySource
from sema.models.target.term import TargetTermDecl
from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.targets.exceptions import CardContentDriftError
from sema.targets.hashing import compute_card_hash
from sema.targets.hashing_utils import canonical_dumps


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def derive_facet_decision(
    facet: Facet,
    annotation: SemanticCompleteness,
    annotation_source: str,
    skip_facets: frozenset[str],
    supplied_by_adapter: bool,
    decided_at: datetime,
) -> FacetDecision:
    if supplied_by_adapter:
        return FacetDecision(
            status=EnrichmentStatus.supplied_by_adapter,
            reason=f"adapter inlined data for facet ({annotation_source})",
            decided_at=decided_at,
        )
    if annotation is SemanticCompleteness.COMPLETE:
        return FacetDecision(
            status=EnrichmentStatus.not_required,
            reason="adapter declared facet COMPLETE",
            decided_at=decided_at,
        )
    if annotation is SemanticCompleteness.EXTERNAL:
        return FacetDecision(
            status=EnrichmentStatus.not_required,
            reason="adapter declared facet EXTERNAL",
            decided_at=decided_at,
        )
    if facet.value in skip_facets:
        return FacetDecision(
            status=EnrichmentStatus.required_skipped,
            reason=f"operator opt-out via build-config skip_facets={sorted(skip_facets)}",
            decided_at=decided_at,
        )
    return FacetDecision(
        status=EnrichmentStatus.required_deferred,
        reason=f"facet declared {annotation.value}; awaiting target-semantic-enrichment",
        decided_at=decided_at,
    )


def effective_annotation(
    facet: Facet,
    descriptor: TargetModelDescriptor,
    entity: TargetEntityDecl,
) -> tuple[SemanticCompleteness, str]:
    if entity.completeness is not None:
        return getattr(entity.completeness, facet.value), "entity-level annotation"
    return getattr(descriptor.completeness, facet.value), "descriptor-level annotation"


def derive_supplied_by_adapter_flags(
    entity: TargetEntityDecl,
    bindings: list[VocabularyBindingDecl],
    terms: list[TargetTermDecl],
) -> dict[Facet, bool]:
    has_inline_synonyms = any(prop.synonyms for prop in entity.properties)
    binding_vocabs = {
        b.vocabulary.name
        for b in bindings
        if b.entity_ref.qualified_name == entity.ref.qualified_name
    }
    inline_term_vocabs = {
        t.vocabulary.name for t in terms if t.vocabulary.source is VocabularySource.INLINE
    }
    has_inline_terms = bool(binding_vocabs & inline_term_vocabs)
    return {
        Facet.structure: False,
        Facet.obligations: False,
        Facet.vocabulary_bindings: False,
        Facet.semantic_aliases: has_inline_synonyms,
        Facet.terms: has_inline_terms,
    }


def derive_enrichment_record(
    entity: TargetEntityDecl,
    descriptor: TargetModelDescriptor,
    bindings: list[VocabularyBindingDecl],
    terms: list[TargetTermDecl],
    skip_facets: frozenset[str],
    iter_terms_external_vocabs: frozenset[str],
) -> EnrichmentDecisionRecord:
    decided_at = _utcnow()
    supplied = derive_supplied_by_adapter_flags(entity, bindings, terms)
    decisions: dict[Facet, FacetDecision] = {}
    for facet in Facet:
        ann, source = effective_annotation(facet, descriptor, entity)
        ann, source = _apply_iter_terms_external(
            facet, entity, bindings, iter_terms_external_vocabs, ann, source
        )
        decisions[facet] = derive_facet_decision(
            facet, ann, source, skip_facets, supplied[facet], decided_at
        )
    return EnrichmentDecisionRecord(
        entity_ref=entity.ref, decisions=decisions, decided_at=decided_at
    )


def _apply_iter_terms_external(
    facet: Facet,
    entity: TargetEntityDecl,
    bindings: list[VocabularyBindingDecl],
    iter_terms_external_vocabs: frozenset[str],
    ann: SemanticCompleteness,
    source: str,
) -> tuple[SemanticCompleteness, str]:
    if facet is not Facet.terms:
        return ann, source
    entity_bindings = [
        b for b in bindings if b.entity_ref.qualified_name == entity.ref.qualified_name
    ]
    if not entity_bindings:
        return ann, source
    flagged = [b.vocabulary.name for b in entity_bindings if b.vocabulary.name in iter_terms_external_vocabs]
    if flagged:
        return (
            SemanticCompleteness.EXTERNAL,
            f"adapter raised NotImplementedError for vocabularies {sorted(set(flagged))}; "
            f"treated as EXTERNAL",
        )
    return ann, source


def populate_card_hashes(
    cards: Iterable[TargetContextCard],
    persisted_hashes: dict[tuple[str, str, str], str] | None = None,
) -> list[LoadedContextCard]:
    out: list[LoadedContextCard] = []
    persisted = persisted_hashes or {}
    for card in cards:
        digest = compute_card_hash(card)
        key = (
            card.entity_ref.target_model_id,
            card.entity_ref.qualified_name,
            card.card_version,
        )
        previous = persisted.get(key)
        if previous is not None and previous != digest:
            raise CardContentDriftError(
                f"card content drift for target_model_id={key[0]!r} "
                f"entity_ref={key[1]!r} card_version={key[2]!r}: "
                f"previous_hash={previous!r} current_hash={digest!r}; bump card_version"
            )
        out.append(LoadedContextCard.from_target_card(card, digest))
    return out


def aggregate_context_card_version(cards: list[LoadedContextCard]) -> str:
    if len(cards) == 1:
        return cards[0].card_version
    pairs = sorted(
        (c.entity_ref.qualified_name, c.card_version) for c in cards
    )
    payload = canonical_dumps(pairs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def card_versions_dict(cards: list[LoadedContextCard]) -> dict[str, str]:
    return {c.entity_ref.qualified_name: c.card_version for c in cards}


