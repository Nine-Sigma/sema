"""load_target — orchestrates normalize → hash → materialize."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timezone

from sema.models.target.context_card import LoadedContextCard
from sema.models.target.descriptor import TargetModelDescriptor
from sema.models.target.enrichment import EnrichmentDecisionRecord
from sema.models.target.loaded import LoadedTarget
from sema.models.target.normalized import NormalizedTargetModel
from sema.models.target.refs import TargetEntityRef, VocabularyRef, VocabularySource
from sema.targets.base import TargetOntologyAdapter
from sema.targets.hashing import SnapshotHasher
from sema.targets.loader_utils import (
    aggregate_context_card_version,
    card_versions_dict,
    derive_enrichment_record,
    populate_card_hashes,
)
from sema.targets.materializer import (
    GraphWriter,
    StageGuard,
    TargetModelMaterializer,
)
from sema.targets.normalizer import TargetModelNormalizer

StageSpy = Callable[[str], None]


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def load_target(
    adapter: TargetOntologyAdapter,
    *,
    writer: GraphWriter,
    selected_refs: Iterable[TargetEntityRef] | None = None,
    skip_facets: Iterable[str] = (),
    persisted_card_hashes: dict[tuple[str, str, str], str] | None = None,
    stage_spy: StageSpy | None = None,
) -> LoadedTarget:
    spy = stage_spy or (lambda _: None)
    guard = StageGuard()

    spy("normalize_started")
    normalized = TargetModelNormalizer.normalize(adapter, selected_refs)
    guard.transition_to(StageGuard.NORMALIZED)
    spy("normalize_completed")

    spy("hash_started")
    target_schema_snapshot_hash = SnapshotHasher.hash(normalized)
    guard.transition_to(StageGuard.HASHED)
    spy("hash_completed")

    iter_terms_external = _detect_iter_terms_external_vocabs(adapter, normalized)
    decisions = _derive_decisions(normalized, frozenset(skip_facets), iter_terms_external)

    cards_with_hash = populate_card_hashes(
        normalized.context_cards, persisted_card_hashes
    )

    spy("materialize_started")
    TargetModelMaterializer.write(
        normalized,
        target_schema_snapshot_hash,
        writer,
        decisions,
        cards_with_hash=cards_with_hash,
        stage_guard=guard,
    )
    spy("materialize_completed")

    return _build_loaded_target(
        normalized.descriptor, normalized, target_schema_snapshot_hash, decisions, cards_with_hash
    )


def _derive_decisions(
    normalized: NormalizedTargetModel,
    skip_facets: frozenset[str],
    iter_terms_external_vocabs: frozenset[str],
) -> list[EnrichmentDecisionRecord]:
    return [
        derive_enrichment_record(
            entity=entity,
            descriptor=normalized.descriptor,
            bindings=normalized.vocabulary_bindings,
            terms=normalized.terms,
            skip_facets=skip_facets,
            iter_terms_external_vocabs=iter_terms_external_vocabs,
        )
        for entity in normalized.entities
    ]


def _detect_iter_terms_external_vocabs(
    adapter: TargetOntologyAdapter, normalized: NormalizedTargetModel
) -> frozenset[str]:
    vocab_names: set[str] = set()
    seen: set[str] = set()
    for binding in normalized.vocabulary_bindings:
        vocab = binding.vocabulary
        if vocab.source is VocabularySource.EXTERNAL:
            continue
        if vocab.name in seen:
            continue
        seen.add(vocab.name)
        if _adapter_treats_vocab_as_external(adapter, vocab):
            vocab_names.add(vocab.name)
    return frozenset(vocab_names)


def _adapter_treats_vocab_as_external(
    adapter: TargetOntologyAdapter, vocab: VocabularyRef
) -> bool:
    iter_terms = getattr(adapter, "iter_terms", None)
    if iter_terms is None:
        return True
    try:
        next(iter(iter_terms(vocab)), None)
    except NotImplementedError:
        return True
    return False


def _build_loaded_target(
    descriptor: TargetModelDescriptor,
    normalized: NormalizedTargetModel,
    target_schema_snapshot_hash: str,
    decisions: list[EnrichmentDecisionRecord],
    cards_with_hash: list[LoadedContextCard],
) -> LoadedTarget:
    card_hashes = {c.entity_ref.qualified_name: c.card_hash for c in cards_with_hash}
    return LoadedTarget(
        descriptor=descriptor,
        target_schema_snapshot_hash=target_schema_snapshot_hash,
        entity_refs=[e.ref for e in normalized.entities],
        enrichment_decisions=decisions,
        card_versions=card_versions_dict(cards_with_hash),
        aggregate_context_card_version=aggregate_context_card_version(cards_with_hash),
        context_cards=cards_with_hash,
        card_hashes=card_hashes,
        materialized_at=_utcnow(),
    )


__all__ = ["load_target"]
