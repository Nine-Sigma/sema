"""TargetModelNormalizer — adapter output → NormalizedTargetModel."""

from __future__ import annotations

from collections.abc import Iterable

from sema.models.target.entity import TargetEntityDecl
from sema.models.target.normalized import NormalizedTargetModel
from sema.models.target.obligation import TargetObligationDecl
from sema.models.target.refs import TargetEntityRef, TargetPropertyRef, VocabularyRef
from sema.models.target.term import TargetTermDecl
from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.targets.base import TargetOntologyAdapter
from sema.targets.normalizer_utils import (
    normalize_entity,
    sort_bindings,
    sort_context_cards,
    sort_entities,
    sort_obligations,
    sort_terms,
    validate_endpoint_targets,
    validate_foreign_keys,
    validate_obligation_required_fields,
    validate_vocabulary_bindings,
)


class TargetModelNormalizer:
    """Validates DTOs, resolves cross-refs, sorts collections."""

    @staticmethod
    def normalize(
        adapter: TargetOntologyAdapter,
        selected_refs: Iterable[TargetEntityRef] | None = None,
    ) -> NormalizedTargetModel:
        descriptor = adapter.describe()
        refs = _select_refs(adapter, selected_refs)
        raw_entities = [adapter.load_entity(ref) for ref in refs]
        entities = [normalize_entity(e) for e in raw_entities]
        obligations = [adapter.load_obligation(ref) for ref in refs]
        bindings = _collect_bindings(adapter, entities)
        terms = _collect_terms(adapter, bindings)
        cards = [adapter.load_context_card(ref) for ref in refs]
        _resolve_cross_refs(entities, obligations, bindings, terms)
        return NormalizedTargetModel(
            descriptor=descriptor,
            entities=sort_entities(entities),
            obligations=sort_obligations(obligations),
            vocabularies=_collect_vocabularies(bindings, terms),
            vocabulary_bindings=sort_bindings(bindings),
            terms=sort_terms(terms),
            context_cards=sort_context_cards(cards),
        )


def _select_refs(
    adapter: TargetOntologyAdapter,
    selected_refs: Iterable[TargetEntityRef] | None,
) -> list[TargetEntityRef]:
    if selected_refs is None:
        return list(adapter.discover_entities())
    return list(selected_refs)


def _collect_bindings(
    adapter: TargetOntologyAdapter, entities: list[TargetEntityDecl]
) -> list[VocabularyBindingDecl]:
    bindings: list[VocabularyBindingDecl] = []
    for entity in entities:
        for prop in entity.properties:
            ref = TargetPropertyRef(entity_ref=entity.ref, property_name=prop.name)
            bindings.extend(adapter.load_vocabulary_bindings(ref))
    return bindings


def _collect_terms(
    adapter: TargetOntologyAdapter, bindings: list[VocabularyBindingDecl]
) -> list[TargetTermDecl]:
    terms: list[TargetTermDecl] = []
    seen: set[str] = set()
    iter_terms = getattr(adapter, "iter_terms", None)
    if iter_terms is None:
        return terms
    for binding in bindings:
        vocab = binding.vocabulary
        if vocab.name in seen:
            continue
        seen.add(vocab.name)
        try:
            terms.extend(iter_terms(vocab))
        except NotImplementedError:
            continue
    return terms


def _collect_vocabularies(
    bindings: list[VocabularyBindingDecl], terms: list[TargetTermDecl]
) -> list[VocabularyRef]:
    seen: dict[str, VocabularyRef] = {}
    for binding in bindings:
        seen.setdefault(binding.vocabulary.name, binding.vocabulary)
    for term in terms:
        seen.setdefault(term.vocabulary.name, term.vocabulary)
    return sorted(seen.values(), key=lambda v: v.name)


def _resolve_cross_refs(
    entities: list[TargetEntityDecl],
    obligations: list[TargetObligationDecl],
    bindings: list[VocabularyBindingDecl],
    terms: list[TargetTermDecl],
) -> None:
    by_qualified = {e.ref.qualified_name: e for e in entities}
    properties_by_entity = {e.ref.qualified_name: list(e.properties) for e in entities}
    for entity in entities:
        validate_endpoint_targets(entity, by_qualified)
    for obligation in obligations:
        validate_obligation_required_fields(obligation, properties_by_entity)
        validate_foreign_keys(obligation, by_qualified)
    validate_vocabulary_bindings(bindings, terms)


__all__ = ["TargetModelNormalizer"]
