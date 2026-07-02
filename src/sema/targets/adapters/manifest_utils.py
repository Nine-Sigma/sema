"""Manifest → target DTO conversion helpers."""

from __future__ import annotations

from sema.models.planner._enums import (
    PrimaryKeyStrategy,
    TargetArtifactKind,
)
from sema.models.planner.target_model import (
    DomainConstraint,
    ExternalSequenceMappingTable,
    FieldEquality,
    FieldPresence,
    ForeignKeyObligation,
    RowPredicate,
)
from sema.models.target.completeness import (
    SemanticCompleteness,
    SemanticCompletenessAnnotations,
)
from sema.models.target.context_card import TargetContextCard
from sema.models.target.endpoints import EdgeEndpointDecl, EdgeEndpointsDecl
from sema.models.target.entity import TargetEntityDecl
from sema.models.target.obligation import TargetObligationDecl
from sema.models.target.properties import TargetPropertyDecl
from sema.models.target.refs import TargetEntityRef, VocabularyRef, VocabularySource
from sema.models.target.term import TargetTermDecl
from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.targets.adapters.manifest_models import (
    ManifestEndpoint,
    ManifestEntity,
    ManifestObligation,
    ManifestProperty,
    ManifestRowClause,
    ManifestRowPredicate,
    ManifestTerm,
    ManifestVocabulary,
    ManifestVocabularyBinding,
    ParsedManifest,
)


_DESCRIPTOR_DEFAULT_COMPLETENESS = SemanticCompletenessAnnotations(
    structure=SemanticCompleteness.COMPLETE,
    obligations=SemanticCompleteness.COMPLETE,
    vocabulary_bindings=SemanticCompleteness.PARTIAL,
    semantic_aliases=SemanticCompleteness.PARTIAL,
    terms=SemanticCompleteness.EXTERNAL,
)


def descriptor_completeness(parsed: ParsedManifest) -> SemanticCompletenessAnnotations:
    if parsed.descriptor.completeness is not None:
        return parsed.descriptor.completeness
    return _DESCRIPTOR_DEFAULT_COMPLETENESS


def vocab_source_index(parsed: ParsedManifest) -> dict[str, VocabularySource]:
    return {v.name: v.source for v in parsed.vocabularies}


def inline_term_vocab_names(parsed: ParsedManifest) -> set[str]:
    names: set[str] = set()
    for term in parsed.terms:
        names.add(term.vocabulary)
    return names


def has_any_inline_term_for_entity_bindings(
    entity: ManifestEntity, inline_vocab_names: set[str]
) -> bool:
    for prop in entity.properties:
        if prop.vocabulary_binding is None:
            continue
        if prop.vocabulary_binding.vocabulary in inline_vocab_names:
            return True
    return False


def effective_entity_completeness(
    entity: ManifestEntity,
    descriptor_default: SemanticCompletenessAnnotations,
    inline_vocab_names: set[str],
) -> SemanticCompletenessAnnotations | None:
    base = entity.completeness or descriptor_default
    semantic_aliases = base.semantic_aliases
    terms = base.terms
    if any(prop.synonyms for prop in entity.properties):
        semantic_aliases = SemanticCompleteness.COMPLETE
    if has_any_inline_term_for_entity_bindings(entity, inline_vocab_names):
        terms = SemanticCompleteness.COMPLETE
    if entity.completeness is None and semantic_aliases is base.semantic_aliases and terms is base.terms:
        return None
    return SemanticCompletenessAnnotations(
        structure=base.structure,
        obligations=base.obligations,
        vocabulary_bindings=base.vocabulary_bindings,
        semantic_aliases=semantic_aliases,
        terms=terms,
    )


def to_target_entity_ref(entity: ManifestEntity, target_model_id: str) -> TargetEntityRef:
    return TargetEntityRef(
        target_model_id=target_model_id,
        qualified_name=entity.qualified_name,
        kind=entity.kind,
    )


def to_target_property(prop: ManifestProperty) -> TargetPropertyDecl:
    return TargetPropertyDecl(
        name=prop.name,
        type=prop.type,
        nullable=prop.nullable,
        synonyms=list(prop.synonyms),
        decoded_values=dict(prop.decoded_values),
        vocabulary_binding=(
            prop.vocabulary_binding.vocabulary
            if prop.vocabulary_binding is not None
            else None
        ),
    )


def to_endpoints_decl(
    parsed: ParsedManifest,
    entity: ManifestEntity,
    target_model_id: str,
) -> EdgeEndpointsDecl | None:
    if entity.endpoints is None:
        return None
    return EdgeEndpointsDecl(
        subject=_to_endpoint_decl("subject", entity.endpoints.subject, parsed, target_model_id),
        object=_to_endpoint_decl("object", entity.endpoints.object, parsed, target_model_id),
    )


def _to_endpoint_decl(
    role: str,
    endpoint: ManifestEndpoint,
    parsed: ParsedManifest,
    target_model_id: str,
) -> EdgeEndpointDecl:
    target_kind = _resolve_endpoint_target_kind(endpoint.target_entity, parsed)
    target_ref = TargetEntityRef(
        target_model_id=target_model_id,
        qualified_name=endpoint.target_entity,
        kind=target_kind,
    )
    return EdgeEndpointDecl(
        role=role,  # type: ignore[arg-type]
        target_entity=target_ref,
        cardinality=endpoint.cardinality,
        nullable=endpoint.nullable,
    )


def _resolve_endpoint_target_kind(
    qualified_name: str, parsed: ParsedManifest
) -> TargetArtifactKind:
    for entity in parsed.entities:
        if entity.qualified_name == qualified_name:
            return entity.kind
    return TargetArtifactKind.GRAPH_NODE


def to_target_entity(
    parsed: ParsedManifest,
    entity: ManifestEntity,
    target_model_id: str,
    descriptor_default: SemanticCompletenessAnnotations,
    inline_vocab_names: set[str],
) -> TargetEntityDecl:
    completeness = effective_entity_completeness(
        entity, descriptor_default, inline_vocab_names
    )
    return TargetEntityDecl(
        ref=to_target_entity_ref(entity, target_model_id),
        properties=[to_target_property(p) for p in entity.properties],
        completeness=completeness,
        endpoints=to_endpoints_decl(parsed, entity, target_model_id),
    )


def to_obligation(
    entity: ManifestEntity,
) -> TargetObligationDecl:
    raw = entity.obligation
    if raw is None:
        return _default_obligation_for_kind(entity)
    minimum = _row_predicate(raw.minimum_viable_row) if raw.minimum_viable_row else _default_minimum(entity)
    external = (
        ExternalSequenceMappingTable(
            mapping_table_name=raw.external_sequence.mapping_table_name,
            canonical_identity_column=raw.external_sequence.canonical_identity_column,
            sequence_column=raw.external_sequence.sequence_column,
        )
        if raw.external_sequence is not None
        else None
    )
    return TargetObligationDecl(
        target_entity=entity.qualified_name,
        required_fields=list(raw.required_fields),
        nullable_fields=list(raw.nullable_fields),
        primary_key=raw.primary_key,
        external_sequence=external,
        foreign_keys=[
            ForeignKeyObligation(
                referenced_entity=fk.referenced_entity,
                join_keys=list(fk.join_keys),
                same_build_required=fk.same_build_required,
            )
            for fk in raw.foreign_keys
        ],
        domain_constraints=[
            DomainConstraint(property_name=dc.property_name, domain_id=dc.domain_id)
            for dc in raw.domain_constraints
        ],
        allowed_defaults=dict(raw.allowed_defaults),
        minimum_viable_row=minimum,
    )


def _default_obligation_for_kind(entity: ManifestEntity) -> TargetObligationDecl:
    if entity.kind is TargetArtifactKind.GRAPH_EDGE:
        return TargetObligationDecl(
            target_entity=entity.qualified_name,
            required_fields=["subject", "object"],
            primary_key=PrimaryKeyStrategy.NATURAL_KEY,
            minimum_viable_row=_default_minimum(entity),
        )
    return TargetObligationDecl(
        target_entity=entity.qualified_name,
        required_fields=[entity.properties[0].name] if entity.properties else ["__placeholder__"],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
    )


def _default_minimum(entity: ManifestEntity) -> RowPredicate | None:
    if entity.kind is TargetArtifactKind.GRAPH_EDGE:
        return RowPredicate(
            op="AND",
            clauses=[
                FieldPresence(field="subject"),
                FieldPresence(field="object"),
            ],
        )
    return None


def _row_predicate(raw: ManifestRowPredicate) -> RowPredicate:
    return RowPredicate(
        op=raw.op, clauses=[_row_clause(c) for c in raw.clauses]
    )


def _row_clause(raw: ManifestRowClause) -> FieldPresence | FieldEquality:
    if raw.kind == "presence":
        return FieldPresence(field=raw.field)
    return FieldEquality(field=raw.field, value=raw.value)


def to_vocabulary_bindings(
    parsed: ParsedManifest,
    entity: ManifestEntity,
    prop: ManifestProperty,
    target_model_id: str,
) -> list[VocabularyBindingDecl]:
    if prop.vocabulary_binding is None:
        return []
    binding = prop.vocabulary_binding
    source = vocab_source_index(parsed).get(binding.vocabulary, VocabularySource.INLINE)
    return [
        VocabularyBindingDecl(
            entity_ref=to_target_entity_ref(entity, target_model_id),
            property_name=prop.name,
            vocabulary=VocabularyRef(name=binding.vocabulary, source=source),
            domain=binding.domain,
            require_standard=binding.require_standard,
            allow_zero_default=binding.allow_zero_default,
            standard_domain_governed=binding.standard_domain_governed,
            effective_date_ref=binding.effective_date_ref,
            resolver_policy_ref=binding.resolver_policy_ref,
        )
    ]


def to_inline_terms(
    parsed: ParsedManifest, vocabulary_name: str
) -> list[TargetTermDecl]:
    source = vocab_source_index(parsed).get(vocabulary_name, VocabularySource.INLINE)
    return [
        TargetTermDecl(
            vocabulary=VocabularyRef(name=t.vocabulary, source=source),
            code=t.code,
            display=t.display,
            domain=t.domain,
        )
        for t in parsed.terms
        if t.vocabulary == vocabulary_name
    ]


def synthesized_card(
    entity: ManifestEntity, target_model_id: str
) -> TargetContextCard:
    obligation_summary = (
        f"{entity.kind.value} {entity.qualified_name}"
        if entity.obligation is None
        else f"required_fields={entity.obligation.required_fields}"
    )
    return TargetContextCard(
        entity_ref=to_target_entity_ref(entity, target_model_id),
        card_version="0.0.0+synthesized",
        description=f"Auto-generated card for {entity.qualified_name}: {obligation_summary}.",
        examples=[],
        obligation_summary=obligation_summary,
        curated_synonyms=[],
    )


def supplied_card(
    entity: ManifestEntity, target_model_id: str
) -> TargetContextCard:
    raw = entity.context_card
    assert raw is not None
    return TargetContextCard(
        entity_ref=to_target_entity_ref(entity, target_model_id),
        card_version=raw.card_version,
        description=raw.description,
        examples=list(raw.examples),
        obligation_summary=raw.obligation_summary,
        curated_synonyms=list(raw.curated_synonyms),
    )


def filter_inline_term_vocabularies(
    parsed: ParsedManifest, vocab_name: str
) -> list[ManifestTerm]:
    return [t for t in parsed.terms if t.vocabulary == vocab_name]


def filter_vocabularies(parsed: ParsedManifest, name: str) -> ManifestVocabulary | None:
    for vocab in parsed.vocabularies:
        if vocab.name == name:
            return vocab
    return None


__all__ = [
    "descriptor_completeness",
    "vocab_source_index",
    "inline_term_vocab_names",
    "to_target_entity_ref",
    "to_target_property",
    "to_endpoints_decl",
    "to_target_entity",
    "to_obligation",
    "to_vocabulary_bindings",
    "to_inline_terms",
    "synthesized_card",
    "supplied_card",
    "filter_inline_term_vocabularies",
    "filter_vocabularies",
    "ManifestVocabularyBinding",
]
